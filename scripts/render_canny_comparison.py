#!/usr/bin/env python3
"""Render continuous moving-track windows; read video only, never use hardware.

Dependencies: numpy, opencv-python, pillow, imageio-ffmpeg.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision/jetson"))
from canny_candidates import lane_canny
from line_detector_v1_warp import LineDetector
from line_preprocess import extract_lane_candidates


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_video(path, fps):
    stream = imageio_ffmpeg.write_frames(
        str(path), (1920, 640), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "19", "-movflags", "+faststart"],
    )
    stream.send(None)
    return stream


def compose(frame, gray, contrast, canny_mask, diag, index, fps,
            playback_index, window_index, segment, fonts):
    title_font, font, small = fonts
    shown = Image.new("RGB", (1920, 640), (19, 24, 31))
    draw = ImageDraw.Draw(shown)
    draw.text((20, 10), "移动片段｜赛道候选对照（离线回放）", font=title_font, fill="white")
    draw.text((1040, 20), "三段连续窗口，保留丢线、遮挡与原录像冻结", font=font, fill=(200, 210, 220))
    labels = [(0, "原视频（旧录像 HUD）"), (640, "鸟瞰原灰度（max RGB）"),
              (960, "当前 contrast 候选"), (1280, "Canny 原始边缘"),
              (1600, "Canny 填充候选")]
    for x, label in labels:
        draw.text((x + 12, 63), label, font=font,
                  fill=(126, 225, 196) if x == 1600 else "white")
    rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
    shown.paste(Image.fromarray(rgb), (0, 115))
    for x, array in ((640, gray), (960, contrast[1]),
                     (1280, diag["raw_edges"]), (1600, canny_mask)):
        shown.paste(Image.fromarray(array).convert("RGB"), (x, 100))
    draw.text((12, 485), "原视频含 HUD；不重新计算行走指令", font=small, fill=(185, 196, 208))
    draw.text((972, 512), f"阈值 {contrast[2]:.0f} / 白像素 {np.count_nonzero(contrast[1])}",
              font=small, fill=(185, 196, 208))
    draw.text((1292, 512), f"低/高阈值 {diag['canny_low']:.1f}/{diag['canny_high']:.1f}",
              font=small, fill=(185, 196, 208))
    draw.text((1612, 512), f"填充后白像素 {np.count_nonzero(canny_mask)}",
              font=small, fill=(126, 225, 196))
    source_time, play_time = index / fps, playback_index / fps
    draw.text((20, 550),
              f"源录像 {source_time:05.1f}s / 帧 {index}   播放 {play_time:04.1f}s / 65.5s   连续片段 {window_index + 1}/3",
              font=font, fill="white")
    if segment["kind"] == "decoded_recording_freeze":
        status = "原录像冻结：画面与非零 vx 均冻结，不代表实际运动"
        color = (255, 185, 90)
    elif segment["role"] == "stress":
        status = "保留压力场景：停顿转换、遮挡或偏离；实际步态不确定"
        color = (255, 185, 90)
    else:
        status = "持续经过赛道标记；移动由画面推断，旧 HUD 不是运动真值"
        color = (126, 225, 196)
    draw.text((20, 586), status, font=small, fill=color)
    draw.text((975, 586), "原始边缘含胶带双边；填充仅是候选，仍需几何配对", font=small, fill="white")
    return np.asarray(shown)


def find_font(explicit):
    choices = ([explicit] if explicit else []) + [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in choices:
        if path and path.is_file():
            return path
    raise RuntimeError("Chinese font not found; supply --font with a TTF/TTC file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing camera_commands3.avi")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=Path,
                        default=ROOT / "docs/audit_2026-10-09/canny_demo/motion_segments.json")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.segments.read_text(encoding="utf-8"))
    windows = audit["recommended_continuous_renderer_windows"]
    video = args.data_dir / "camera_commands3.avi"
    calibration_path = ROOT / "docs/audit_2026-10-09/line_extraction/annotations.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["meta"]["calibration"]
    source_hashes = {name: sha256(ROOT / "new_vision/jetson" / name) for name in
                     ("canny_candidates.py", "line_preprocess.py", "line_detector_v1_warp.py")}
    font_path = find_font(args.font)
    fonts = tuple(ImageFont.truetype(str(font_path), size) for size in (30, 23, 20))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(prop)) for prop in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != tuple(audit["size"]) or abs(fps - audit["fps"]) > 0.001 or count != audit["n_frames"]:
        cap.release()
        raise RuntimeError("Source metadata differs from the audited segment file")
    detector = LineDetector(width, height, cam_height_cm=calibration["height_cm"],
                            cam_pitch_deg=calibration["pitch_deg"],
                            cam_vfov_deg=calibration["vfov_deg"],
                            z_calib=calibration["z_calib"],
                            lane_width_cm=calibration["lane_width_cm"])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "comparison_moving_65s.mp4"
    stream = write_video(target, fps)
    selected = {565, 590, 1040, 1150}
    gif, summaries = [], []
    playback_index = 0
    try:
        for window_index, window in enumerate(windows):
            cap.set(cv2.CAP_PROP_POS_FRAMES, window["start_frame"])
            for index in range(window["start_frame"], window["end_frame"]):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode selected source frame {index}")
                bird = cv2.warpPerspective(frame, detector.M, (320, 400))
                gray = np.max(bird, axis=2)
                contrast = extract_lane_candidates(gray, "contrast")
                mask, diag = lane_canny(gray)
                segment = next(s for s in audit["segments"]
                               if s["start_frame"] <= index < s["end_frame"])
                panel = compose(frame, gray, contrast, mask, diag, index, fps,
                                playback_index, window_index, segment, fonts)
                stream.send(panel)
                if 510 <= index < 590 and index % 2 == 0:
                    gif.append(Image.fromarray(panel).resize((960, 320), Image.Resampling.LANCZOS))
                if index in selected:
                    Image.fromarray(panel).save(args.output / f"comparison_frame_{index}.png")
                    for name, pixels in (("bird_gray", gray), ("contrast", contrast[1]),
                                         ("canny_edges", diag["raw_edges"]), ("canny_filled", mask)):
                        cv2.imwrite(str(args.output / f"{name}_frame_{index}.png"), pixels)
                summaries.append({"source_frame": index, "source_time_s": index / fps,
                                  "output_frame": playback_index,
                                  "role": segment["role"], "kind": segment["kind"],
                                  "contrast_threshold": contrast[2],
                                  "contrast_pixels": int(np.count_nonzero(contrast[1])),
                                  "canny_low": diag["canny_low"], "canny_high": diag["canny_high"],
                                  "canny_edge_pixels": diag["canny_raw_edge_pixels"],
                                  "canny_filled_pixels": int(np.count_nonzero(mask))})
                playback_index += 1
                if playback_index % 100 == 0:
                    print(f"Rendered {playback_index}/655 selected frames", flush=True)
    finally:
        cap.release()
        stream.close()
    expected = sum(w["end_frame"] - w["start_frame"] for w in windows)
    if playback_index != expected:
        raise RuntimeError(f"Incomplete output: {playback_index}, expected {expected}")
    if gif:
        gif[0].save(args.output / "preview_8s.gif", save_all=True, append_images=gif[1:],
                    duration=round(2000 / fps), loop=0, optimize=False)
    manifest = {
        "source_video": str(video), "source_video_sha256": sha256(video),
        "source_frames": count, "source_size": [width, height], "fps": fps,
        "calibration": calibration, "calibration_annotations_sha256": sha256(calibration_path),
        "comparison_size": [1920, 640], "output_frames": playback_index,
        "output_duration_s": playback_index / fps, "source_modules_sha256": source_hashes,
        "windows": windows, "representative_source_frames": sorted(selected),
        "gif_source_frames": [510, 590], "gif_duration_s": 8,
        "font": str(font_path), "opencv": cv2.__version__,
        "gray_definition": "Unmodified warped source, max of RGB channels, matching existing lane demo",
        "scope": "Image candidates only; no lane pairing, steering simulation, camera/serial/motor access",
        "limitations": [
            "Physical gait inferred from track landmark traversal and bob; HUD speed or optical motion is not gait ground truth.",
            "Existing HUD is recorded historical output, not output of this comparison.",
            "Raw Canny marks both sides of black tape; these sides are not two distinct lane boundaries.",
            "Filled Canny mask reconstructs supported dark strokes only; geometry still must choose genuine lane boundaries.",
            "Failure, cable, person occlusion, off-track and stop transitions are retained within continuous windows.",
            "Identical decoded-frame freeze543-558 is retained and labeled; freeze1178-1212 is outside selected windows.",
            "Fixed45-degree calibration; actual pose, ground truth lane widths, unseen-scene or hardware performance are not validated.",
            "No all-six-card ground truth is established by this video.",
        ],
        "frame_summaries": summaries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "motion_segments.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {playback_index} frames / {playback_index / fps:.1f}s: {target}", flush=True)


if __name__ == "__main__":
    main()
