# app.py — Gradio front-end for ColorMNet with keyframe segment processing
# Processes video in segments based on keyframe ranges
# Each segment uses one keyframe as reference (no copying needed)

import os
import sys
import shutil
import urllib.request
import re
from os import path
import io
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

import gradio as gr
from PIL import Image
import torch

# ----------------- BASIC INFO -----------------
CHECKPOINT_URL = "https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth"
CHECKPOINT_LOCAL = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"

TITLE = "ColorMNet — 批量视频着色 / Batch Video Colorization"
DESC = """
**关键帧命名 / Keyframe Naming:** `keyframe_XXXX_YYYYY.png`
- XXXX = 关键帧编号 / keyframe ID
- YYYYY = 起始帧号 / starting frame number

**自动映射 / Auto Mapping:** `video_X_Y` → `video_X_key_Y`

**处理方式 / Processing:** 按关键帧分段处理，每段使用一个参考帧 / Segmented by keyframes
"""

PAPER = """### ECCV 2024 — ColorMNet | [GitHub](https://github.com/yyang181/colormnet)"""

# ----------------- TEMP WORKDIR -----------------
TEMP_ROOT = path.join(os.getcwd(), "_colormnet_tmp")
INPUT_DIR = "input_video"
REF_DIR = "ref"
OUTPUT_DIR = "output"

def reset_temp_root():
    if path.isdir(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    for sub in (INPUT_DIR, REF_DIR, OUTPUT_DIR):
        os.makedirs(path.join(TEMP_ROOT, sub), exist_ok=True)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def ensure_checkpoint():
    try:
        if not path.exists(CHECKPOINT_LOCAL):
            print(f"[INFO] Downloading checkpoint: {CHECKPOINT_URL}")
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_LOCAL)
    except Exception as e:
        print(f"[WARN] Checkpoint download failed: {e}")

# ----------------- FOLDER MAPPING -----------------
def scan_and_map_folders(input_root: str, ref_root: str):
    """Map video_X_Y -> video_X_key_Y folders"""
    pattern = re.compile(r'^video_(\d+)_(\d+)$')
    mappings = []
    
    if not path.isdir(input_root):
        return mappings
    
    for folder in sorted(os.listdir(input_root)):
        match = pattern.match(folder)
        if match and path.isdir(path.join(input_root, folder)):
            ref_name = f"video_{match.group(1)}_key_{match.group(2)}"
            ref_path = path.join(ref_root, ref_name)
            if path.isdir(ref_path):
                mappings.append((path.join(input_root, folder), ref_path, folder))
    
    return mappings

# ----------------- KEYFRAME PARSING -----------------
def parse_keyframe_filename(filename: str):
    """Parse keyframe_XXXX_YYYYY.png -> (keyframe_id, start_frame, path)"""
    pattern = re.compile(r'^keyframe_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
    match = pattern.match(filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def build_keyframe_segments(keyframe_folder: str, input_folder: str):
    """
    Build list of segments: [(start_frame, end_frame, keyframe_path), ...]
    Each segment defines which input frames use which keyframe.
    """
    # Parse all keyframes
    keyframes = []
    for f in os.listdir(keyframe_folder):
        parsed = parse_keyframe_filename(f)
        if parsed:
            kf_id, start_frame = parsed
            keyframes.append((start_frame, path.join(keyframe_folder, f), kf_id))
    
    if not keyframes:
        return []
    
    # Sort by start_frame
    keyframes.sort(key=lambda x: x[0])
    
    # Get input frame count
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])
    if not input_files:
        return []
    
    # Extract max frame number from input filenames
    max_frame = 0
    for f in input_files:
        match = re.match(r'^(\d+)', f)
        if match:
            max_frame = max(max_frame, int(match.group(1)))
    
    if max_frame == 0:
        max_frame = len(input_files)
    
    # Build segments
    segments = []
    for i, (start, kf_path, kf_id) in enumerate(keyframes):
        if i + 1 < len(keyframes):
            end = keyframes[i + 1][0] - 1
        else:
            end = max_frame
        segments.append((start, end, kf_path, kf_id))
    
    return segments

def get_input_files_for_segment(input_folder: str, start_frame: int, end_frame: int):
    """Get list of (filename, frame_number) for frames in range [start, end]"""
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = []
    
    for f in sorted(os.listdir(input_folder)):
        if f.lower().endswith(valid_ext):
            match = re.match(r'^(\d+)', f)
            if match:
                frame_num = int(match.group(1))
                if start_frame <= frame_num <= end_frame:
                    files.append((f, frame_num))
    
    return sorted(files, key=lambda x: x[1])

# ----------------- CLI MAPPING -----------------
CONFIG_TO_CLI = {
    "FirstFrameIsNotExemplar": "--FirstFrameIsNotExemplar",
    "dataset": "--dataset", "split": "--split",
    "save_all": "--save_all", "benchmark": "--benchmark",
    "disable_long_term": "--disable_long_term",
    "max_mid_term_frames": "--max_mid_term_frames",
    "min_mid_term_frames": "--min_mid_term_frames",
    "max_long_term_elements": "--max_long_term_elements",
    "num_prototypes": "--num_prototypes",
    "top_k": "--top_k", "mem_every": "--mem_every",
    "deep_update_every": "--deep_update_every",
    "save_scores": "--save_scores", "flip": "--flip",
    "size": "--size", "reverse": "--reverse",
}

def build_args_list(d16_batch_path: str, out_path: str, ref_root: str, cfg: dict):
    args = ["--d16_batch_path", d16_batch_path, "--ref_path", ref_root, "--output", out_path]
    for k, v in cfg.items():
        if k in CONFIG_TO_CLI:
            if isinstance(v, bool) and v:
                args.append(CONFIG_TO_CLI[k])
            elif v is not None and not isinstance(v, bool):
                args.extend([CONFIG_TO_CLI[k], str(v)])
    return args

# ----------------- SEGMENT PROCESSING -----------------
def process_segment(segment_name: str, input_files: list, input_folder: str,
                    keyframe_path: str, output_folder: str, user_config: dict,
                    test_module, log_lines: list, debug: bool):
    """
    Process one keyframe segment.
    - input_files: list of (filename, frame_num)
    - keyframe_path: path to the keyframe image to use as reference
    - output_folder: where to save colorized images (with original names)
    """
    if not input_files:
        return 0
    
    reset_temp_root()
    
    temp_input = path.join(TEMP_ROOT, INPUT_DIR, segment_name)
    temp_ref = path.join(TEMP_ROOT, REF_DIR, segment_name)
    temp_output = path.join(TEMP_ROOT, OUTPUT_DIR, segment_name)
    
    for d in (temp_input, temp_ref, temp_output):
        ensure_dir(d)
    
    # Copy input images with sequential naming
    idx_to_original = {}
    for idx, (filename, frame_num) in enumerate(input_files):
        src = path.join(input_folder, filename)
        dst = path.join(temp_input, f"{idx:05d}.png")
        img = Image.open(src)
        img.save(dst, "PNG")
        idx_to_original[idx] = filename
    
    # Create single reference (just copy keyframe as 00000.png)
    kf_dst = path.join(temp_ref, "00000.png")
    kf_img = Image.open(keyframe_path)
    kf_img.save(kf_dst, "PNG")
    
    # Run inference
    args_list = build_args_list(
        d16_batch_path=path.join(TEMP_ROOT, INPUT_DIR),
        out_path=path.join(TEMP_ROOT, OUTPUT_DIR),
        ref_root=path.join(TEMP_ROOT, REF_DIR),
        cfg=user_config
    )
    
    if debug:
        log_lines.append(f"    Args: {' '.join(args_list)}")
    
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            entry = getattr(test_module, "run_cli", None)
            if entry is None:
                raise RuntimeError("test.py missing run_cli")
            entry(args_list)
        if debug:
            log_lines.append(buf.getvalue())
    except Exception as e:
        log_lines.append(f"    ERROR: {e}")
        return 0
    
    # Copy output with original filenames
    ensure_dir(output_folder)
    copied = 0
    output_files = sorted([f for f in os.listdir(temp_output) if f.endswith('.png')])
    
    for idx, out_file in enumerate(output_files):
        if idx in idx_to_original:
            orig_base = path.splitext(idx_to_original[idx])[0]
            dst_name = f"{orig_base}.png"
        else:
            dst_name = out_file
        
        src = path.join(temp_output, out_file)
        dst = path.join(output_folder, dst_name)
        shutil.copy2(src, dst)
        copied += 1
    
    return copied

# ----------------- GRADIO HANDLER -----------------
def gradio_infer(
    debug_shapes, gpu_id,
    input_root_path, ref_root_path, output_root_path,
    first_not_exemplar, dataset, split, save_all, benchmark,
    disable_long_term, max_mid, min_mid, max_long,
    num_proto, top_k, mem_every, deep_update,
    save_scores, flip, size, reverse,
    progress=gr.Progress()
):
    # GPU setup
    gpu_id = gpu_id or 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Validate inputs
    for label, val in [("输入目录", input_root_path), ("参考目录", ref_root_path), ("输出目录", output_root_path)]:
        if not val or not val.strip():
            return f"请输入{label} / Please enter {label}"
    
    input_root_path = input_root_path.strip()
    ref_root_path = ref_root_path.strip()
    output_root_path = output_root_path.strip()
    
    for label, p in [("输入", input_root_path), ("参考", ref_root_path)]:
        if not path.isdir(p):
            return f"{label}目录不存在 / {label} folder not found: {p}"
    
    ensure_dir(output_root_path)
    
    # Scan folders
    mappings = scan_and_map_folders(input_root_path, ref_root_path)
    if not mappings:
        return "未找到匹配文件夹 / No matching folders (video_X_Y -> video_X_key_Y)"
    
    log_lines = [
        f"GPU_ID={gpu_id}",
        "=" * 60,
        "📁 FOLDER MAPPING",
        "=" * 60,
    ]
    for inp, ref, name in mappings:
        log_lines.append(f"  {name} → {path.basename(ref)}")
    log_lines.append("=" * 60)
    log_lines.append("")
    
    # Config
    user_config = {
        "FirstFrameIsNotExemplar": bool(first_not_exemplar) if first_not_exemplar is not None else True,
        "dataset": dataset or "D16_batch",
        "split": split or "val",
        "save_all": save_all if save_all is not None else True,
        "benchmark": benchmark or False,
        "disable_long_term": disable_long_term or False,
        "max_mid_term_frames": int(max_mid) if max_mid else 10,
        "min_mid_term_frames": int(min_mid) if min_mid else 5,
        "max_long_term_elements": int(max_long) if max_long else 10000,
        "num_prototypes": int(num_proto) if num_proto else 128,
        "top_k": int(top_k) if top_k else 30,
        "mem_every": int(mem_every) if mem_every else 5,
        "deep_update_every": int(deep_update) if deep_update else -1,
        "save_scores": save_scores or False,
        "flip": flip or False,
        "size": int(size) if size else -1,
        "reverse": reverse or False,
    }

    ensure_checkpoint()

    try:
        import test_app as test
    except Exception as e:
        return f"Failed to import test_app: {e}"

    total_images = 0
    total_folders = 0
    
    for folder_idx, (input_folder, ref_folder, folder_name) in enumerate(mappings):
        progress((folder_idx, len(mappings)), desc=f"📂 {folder_name}")
        log_lines.append(f"[{folder_idx+1}/{len(mappings)}] 📂 {folder_name}")
        
        # Build keyframe segments
        segments = build_keyframe_segments(ref_folder, input_folder)
        if not segments:
            log_lines.append(f"  ⚠️ No keyframes found in {ref_folder}")
            continue
        
        log_lines.append(f"  Found {len(segments)} keyframe segments:")
        for start, end, kf_path, kf_id in segments:
            log_lines.append(f"    📷 keyframe_{kf_id:04d}: frames {start}-{end}")
        
        output_folder = path.join(output_root_path, folder_name)
        folder_images = 0
        
        # Process each segment
        for seg_idx, (start, end, kf_path, kf_id) in enumerate(segments):
            segment_name = f"{folder_name}_seg{seg_idx}"
            input_files = get_input_files_for_segment(input_folder, start, end)
            
            if not input_files:
                log_lines.append(f"    ⚠️ No frames for segment {start}-{end}")
                continue
            
            log_lines.append(f"  Processing segment {seg_idx+1}/{len(segments)}: {len(input_files)} frames with keyframe_{kf_id:04d}")
            
            copied = process_segment(
                segment_name, input_files, input_folder,
                kf_path, output_folder, user_config,
                test, log_lines, debug_shapes
            )
            
            folder_images += copied
            log_lines.append(f"    ✅ Colorized {copied} frames")
            
            # Clear CUDA
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except:
                pass
        
        total_images += folder_images
        total_folders += 1
        log_lines.append(f"  📁 {folder_name}: {folder_images} images saved to {output_folder}")
        log_lines.append("")

    log_lines.append("=" * 60)
    log_lines.append(f"✅ 完成 / Done")
    log_lines.append(f"📁 Folders: {total_folders}/{len(mappings)}")
    log_lines.append(f"🖼️ Total images: {total_images}")
    log_lines.append(f"📂 Output: {output_root_path}")
    
    return "\n".join(log_lines)

# ----------------- UI -----------------
with gr.Blocks() as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(PAPER)
    gr.Markdown(DESC)

    with gr.Row():
        gpu_id = gr.Number(label="GPU ID", value=0, precision=0)
        debug_shapes = gr.Checkbox(label="Debug Logs", value=False)

    with gr.Row():
        inp_root = gr.Textbox(label="Input Root", placeholder="/path/to/input (video_1_1, ...)")
        ref_root = gr.Textbox(label="Reference Root", placeholder="/path/to/ref (video_1_key_1, ...)")
        out_root = gr.Textbox(label="Output", placeholder="/path/to/output")

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            first_not_exemplar = gr.Checkbox(label="FirstFrameIsNotExemplar", value=True)
            reverse = gr.Checkbox(label="reverse", value=False)
            dataset = gr.Textbox(label="dataset", value="D16_batch")
            split = gr.Textbox(label="split", value="val")
            save_all = gr.Checkbox(label="save_all", value=True)
            benchmark = gr.Checkbox(label="benchmark", value=False)
        with gr.Row():
            disable_long_term = gr.Checkbox(label="disable_long_term", value=False)
            max_mid = gr.Number(label="max_mid_term_frames", value=10, precision=0)
            min_mid = gr.Number(label="min_mid_term_frames", value=5, precision=0)
            max_long = gr.Number(label="max_long_term_elements", value=10000, precision=0)
            num_proto = gr.Number(label="num_prototypes", value=128, precision=0)
        with gr.Row():
            top_k = gr.Number(label="top_k", value=30, precision=0)
            mem_every = gr.Number(label="mem_every", value=5, precision=0)
            deep_update = gr.Number(label="deep_update_every", value=-1, precision=0)
            save_scores = gr.Checkbox(label="save_scores", value=False)
            flip = gr.Checkbox(label="flip", value=False)
            size = gr.Number(label="size", value=-1, precision=0)

    run_btn = gr.Button("🎨 Start Coloring", variant="primary")
    status = gr.Textbox(label="Logs", interactive=False, lines=30)

    run_btn.click(
        fn=gradio_infer,
        inputs=[
            debug_shapes, gpu_id,
            inp_root, ref_root, out_root,
            first_not_exemplar, dataset, split, save_all, benchmark,
            disable_long_term, max_mid, min_mid, max_long,
            num_proto, top_k, mem_every, deep_update,
            save_scores, flip, size, reverse
        ],
        outputs=[status]
    )

if __name__ == "__main__":
    ensure_checkpoint()
    demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860, share=False)
