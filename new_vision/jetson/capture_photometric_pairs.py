#!/usr/bin/env python3
"""Interactively capture same-scene auto and fixed photometric calibration pairs.

Keep the camera, grayscale board, and lighting fixed. This tool does not start
the vision controller, gait policy, serial link, or motors.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from photometric_capture import (CalibrationError, CameraReader, CaptureStabilityConfig,
                                 V4L2Controller, atomic_json, capture_group,
                                 create_session_directory, utc_now)


def _input_float(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("请输入数字。")


def _input_int(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return int(value)
        except ValueError:
            print("请输入整数。")


def _camera_configuration(args):
    from camera_config import load as load_camera
    config = load_camera(args.camera_profile)
    index = args.camera_index if args.camera_index is not None else int(config.get("index", 0))
    width = args.width if args.width is not None else int(config.get("width", 1280))
    height = args.height if args.height is not None else int(config.get("height", 720))
    index_source = "command line" if args.camera_index is not None else f"camera_config profile={config.get('profile')}"
    resolution_source = "command line" if args.width is not None or args.height is not None else f"camera_config profile={config.get('profile')}"
    print(f"[camera] index={index} source={index_source}; requested={width}x{height} source={resolution_source}")
    return index, width, height, config


def _fixed_values(args):
    exp = args.fixed_exposure_ms
    exp_source = "command line" if exp is not None else "terminal prompt"
    if exp is None:
        if not sys.stdin.isatty():
            raise CalibrationError("provide --fixed-exposure-ms in non-interactive mode")
        exp = _input_float("输入固定曝光时间（ms；不预设数值）：")
    if not math.isfinite(float(exp)) or exp <= 0:
        raise CalibrationError("固定曝光必须是正数")
    print(f"[fixed settings] exposure={exp:g} ms source={exp_source}; "
          "white_balance=auto (both groups)")
    return float(exp), None, {"exposure_ms": exp_source, "white_balance": "auto (both groups)"}


def _format_fourcc(value):
    if not math.isfinite(float(value)):
        return "unknown"
    value = int(value)
    return "".join(chr((value >> (8*i)) & 0xff) for i in range(4)).rstrip("\x00") or "unknown"


def _metadata_callback(metadata_path, root_metadata):
    def update(group_result):
        root_metadata.setdefault("groups", {})[group_result["mode"]] = group_result
        atomic_json(metadata_path, root_metadata)
    return update


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=None,
                        help="Windows DirectShow / Linux V4L2 index; defaults to camera_config.py")
    parser.add_argument("--camera-profile", default=None,
                        help="camera_config.py profile; defaults to its active profile")
    parser.add_argument("--width", type=int, default=None, help="requested decoded frame width")
    parser.add_argument("--height", type=int, default=None, help="requested decoded frame height")
    parser.add_argument("--output-dir", default="calibration",
                        help="session root; every run creates a new timestamped directory")
    parser.add_argument("--fixed-exposure-ms", type=float, default=None,
                        help="fixed exposure in ms; required or prompted, never guessed")
    parser.add_argument("--allow-exposure-rounding", action="store_true",
                        help="Windows only: explicitly allow nearest supported exposure step; "
                             "requested and driver-read exposure are both recorded")
    parser.add_argument("--warmup-seconds", type=float, default=3.0,
                        help="minimum discarded transition time (must be >=3 seconds)")
    parser.add_argument("--stability-window-seconds", type=float, default=1.5)
    parser.add_argument("--stability-timeout-seconds", type=float, default=25.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)

    print("固定相机、标定板和灯光，两组之间不能移动；")
    print("请先关闭其他占用摄像头的程序。")
    try:
        index, width, height, camera_config = _camera_configuration(args)
        fixed_exp, fixed_wb, fixed_sources = _fixed_values(args)
        if width <= 0 or height <= 0:
            raise CalibrationError("requested width and height must be positive")
        stability = CaptureStabilityConfig(
            warmup_seconds=args.warmup_seconds,
            stability_window_seconds=args.stability_window_seconds,
            stability_timeout_seconds=args.stability_timeout_seconds,
            sample_interval_seconds=args.sample_interval_seconds)
    except (CalibrationError, ValueError) as exc:
        parser.error(str(exc))

    try:
        import cv2
    except ImportError as exc:
        print("错误：相机采集需要安装 OpenCV Python（cv2）；本次未打开摄像头。", file=sys.stderr)
        return 2
    if sys.platform == "win32":
        capture_backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        capture_backend = cv2.CAP_V4L2
    else:
        print("错误：采集支持 Windows DirectShow 或 Linux V4L2。", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(index, capture_backend)
    if not cap.isOpened():
        cap.release()
        print(f"错误：无法打开 camera index {index}；未停止其他服务或进程。", file=sys.stderr)
        return 2
    initial_width_property = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_height_property = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    initial_width = (int(round(initial_width_property))
                     if math.isfinite(initial_width_property) and initial_width_property > 0 else None)
    initial_height = (int(round(initial_height_property))
                      if math.isfinite(initial_height_property) and initial_height_property > 0 else None)
    initial_fourcc = _format_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    initial_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(initial_fps) or initial_fps <= 0:
        initial_fps = "unknown"
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"

    controller = None
    try:
        if sys.platform == "win32":
            from photometric_windows import DirectShowController
            controller = DirectShowController(index, args.allow_exposure_rounding)
        else:
            controller = V4L2Controller(f"/dev/video{index}")
        initial = controller.query_initial_state()
        device = controller.device
        if sys.platform == "win32":
            selection = controller.resolve_fixed_exposure(fixed_exp)
            print(f"[exposure] requested={fixed_exp:g} ms; "
                  f"supported selection={selection['selected_ms']:g} ms "
                  f"(DirectShow log2 seconds={selection['native_value']})")
    except Exception as exc:
        cap.release()
        if controller is not None and hasattr(controller, "close"):
            controller.close()
        print(f"错误：无法查询相机控制菜单/初始值，拒绝拍摄：{exc}", file=sys.stderr)
        return 2
    print(f"[controls] queried {len(initial['controls'])} {backend} controls from {device}.")
    for name, info in initial["controls"].items():
        menu = info.get("menu") or None
        print(f"  {name}: type={info.get('kind')} range="
              f"{info.get('min', '?')}..{info.get('max', '?')} step={info.get('step', '?')} "
              f"current={info.get('actual', 'unknown')} menu={menu}")

    set_width = bool(cap.set(cv2.CAP_PROP_FRAME_WIDTH, width))
    set_height = bool(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height))
    width_readback = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_readback = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_width = int(round(width_readback)) if math.isfinite(width_readback) else None
    actual_height = int(round(height_readback)) if math.isfinite(height_readback) else None
    actual_fourcc = _format_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(actual_fps) or actual_fps <= 0:
        actual_fps = "unknown"
    print(f"[camera] requested={width}x{height} set_returns=({set_width},{set_height}) "
          f"property_readback={actual_width}x{actual_height} format={actual_fourcc} "
          f"fps={actual_fps} backend={backend}")

    try:
        session = create_session_directory(args.output_dir)
    except OSError as exc:
        cap.release()
        if hasattr(controller, "close"):
            controller.close()
        print(f"错误：无法创建新的session目录：{exc}", file=sys.stderr)
        return 2
    metadata_path = session / "metadata.json"
    metadata = {
        "schema_version": 1, "session_id": session.name,
        "started_at": utc_now(), "device": device, "camera_index": index,
        "camera_profile": camera_config.get("profile", "unknown"),
        "camera_profile_values": {k: camera_config.get(k) for k in ("index", "width", "height")},
        "initial_camera_format": {"resolution_property": [initial_width, initial_height],
                                  "format": initial_fourcc, "fps": initial_fps},
        "requested_resolution": [width, height],
        "resolution_set_return_values": {"width": set_width, "height": set_height},
        "actual_resolution": [actual_width, actual_height],
        "format": actual_fourcc, "fps_readback": actual_fps, "backend": backend,
        "saved_image_format": "PNG of unmodified OpenCV-decoded camera frame; not sensor RAW",
        "fixed_settings": {"exposure_ms": fixed_exp, "white_balance": "auto",
                           "source": fixed_sources,
                           "allow_exposure_rounding": args.allow_exposure_rounding},
        "capture_config": asdict(stability),
        "stability_roi": "central x=25..75%, y=35..75%; median brightness and R/G,B/G",
        "initial_camera_state": initial,
        "groups": {}, "restore": None,
    }
    atomic_json(metadata_path, metadata)
    print(f"[session] {session}")
    reader = CameraReader(cap).start()
    controls_safe = True
    try:
        _, first_frame, _ = reader.read_after(-1, timeout=5.0)
        metadata["actual_resolution"] = [int(first_frame.shape[1]), int(first_frame.shape[0])]
        atomic_json(metadata_path, metadata)
        print(f"[camera] first decoded frame confirms {first_frame.shape[1]}x{first_frame.shape[0]}")
        while True:
            print("\n输入 0：自动曝光 + 自动白平衡，预热后保存10张")
            print("输入 1：固定曝光 + 自动白平衡，预热后保存10张")
            print("输入 q：退出并恢复可恢复的初始相机设置")
            try:
                command = input("> ").strip().lower()
            except EOFError:
                command = "q"
            if command == "q":
                break
            if command not in ("0", "1"):
                print("请输入 0、1 或 q。")
                continue
            if not controls_safe:
                print("本session发生控制或取帧/保存失败，已停止后续拍摄；请输入 q 退出。")
                continue
            group = "auto" if command == "0" else "fixed"
            if group in metadata["groups"]:
                print(f"{group} 组已有尝试记录，拒绝覆盖。请退出并启动新 session 重拍。")
                continue

            def progress(info):
                print(f"\r等待稳定：{info['elapsed_seconds']:.1f}s，窗口采样 {info['samples']} 帧",
                      end="", flush=True)

            result = capture_group(
                group, reader, controller, session / group, stability,
                fixed_exposure_ms=fixed_exp, fixed_wb_k=fixed_wb,
                metadata_callback=_metadata_callback(metadata_path, metadata),
                on_progress=progress)
            if result["status"] == "control_failed":
                print(f"\n控制设置/读回失败，不拍照：{result.get('control_error')}")
                rollback = controller.restore_initial()
                result["rollback_to_initial"] = rollback
                metadata["groups"][group] = result
                metadata["last_failure_rollback"] = rollback
                atomic_json(metadata_path, metadata)
                if rollback["status"] != "restored":
                    controls_safe = False
                    print(f"控制回滚状态：{rollback['status']}；请退出，本session不再拍摄。")
                continue
            if result["status"] == "stability_failed":
                print(f"\n相机取帧失败：{result['stability'].get('error', result['stability'].get('reason'))}")
            elif result["status"] == "complete":
                print("\n已保存 10 张 PNG；可以继续拍摄另一组。")
            elif result["status"] == "unqualified_drift":
                print("\n已保存 10 张 PNG；漂移超限仅记录提示，可以继续拍摄另一组。")
                drift = result["drift"]
                for reason in drift["reasons"]:
                    if reason == "brightness_drift":
                        print(f"  亮度波动 {drift['brightness_relative_range']:.2%} "
                              f"> 阈值 {stability.group_brightness_relative_range:.2%}")
                    elif reason in ("r_over_g_drift", "b_over_g_drift"):
                        key = "r_over_g_p90_p10" if reason == "r_over_g_drift" else "b_over_g_p90_p10"
                        label = "R/G" if reason == "r_over_g_drift" else "B/G"
                        print(f"  {label} 比值波动 {drift[key]:.4f} "
                              f"> 阈值 {stability.group_channel_ratio_range:.4f}")
                    else:
                        print(f"  相机参数漂移：{reason}；实际范围：{drift['parameter_ranges']}")
            else:
                print(f"\n本组未完成：{result.get('capture_error', result['status'])}；请退出并用新session重拍。")
            if result["status"] not in ("complete", "unqualified_drift"):
                # Keep the evidence and status, but never append a second set to
                # the same ten-frame calibration group.
                controls_safe = False
                print("本组未完整拍摄；请输入 q 退出，然后用新session重拍。")
        return_code = 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，开始恢复相机设置。")
        return_code = 130
    except CalibrationError as exc:
        print(f"\n相机读取失败：{exc}；没有继续拍摄。", file=sys.stderr)
        return_code = 2
    finally:
        restore = controller.restore_initial()
        reader_stopped = reader.stop(timeout=3.0)
        if not reader_stopped:
            resolution_restore = {"status": "not_restored_camera_reader_still_active"}
        elif initial_width is None or initial_height is None:
            resolution_restore = {"status": "unknown_initial_resolution",
                                  "initial_property": [initial_width, initial_height]}
        else:
            set_old_width = bool(cap.set(cv2.CAP_PROP_FRAME_WIDTH, initial_width))
            set_old_height = bool(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, initial_height))
            restored_width_raw = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            restored_height_raw = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            restored_width = (int(round(restored_width_raw))
                              if math.isfinite(restored_width_raw) else None)
            restored_height = (int(round(restored_height_raw))
                               if math.isfinite(restored_height_raw) else None)
            restored_fourcc = _format_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            restored_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(restored_fps) or restored_fps <= 0:
                restored_fps = "unknown"
            resolution_ok = (restored_width == initial_width and restored_height == initial_height)
            format_ok = (initial_fourcc == "unknown" or restored_fourcc == initial_fourcc)
            fps_ok = (initial_fps == "unknown" or restored_fps == "unknown"
                      or math.isclose(float(initial_fps),float(restored_fps),rel_tol=.01,abs_tol=.1))
            resolution_restore = {"status": "restored" if resolution_ok and format_ok and fps_ok
                                  else "readback_mismatch",
                                  "initial": [initial_width, initial_height],
                                  "actual": [restored_width, restored_height],
                                  "initial_format": initial_fourcc,
                                  "actual_format": restored_fourcc,
                                  "initial_fps": initial_fps, "actual_fps": restored_fps,
                                  "set_return_values": {"width": set_old_width,
                                                        "height": set_old_height}}
        restore["resolution"] = resolution_restore
        if resolution_restore["status"] != "restored" and restore["status"] == "restored":
            restore["status"] = "partial"
        metadata["restore"] = restore
        metadata["finished_at"] = utc_now()
        try:
            atomic_json(metadata_path, metadata)
        finally:
            try:
                reader.close()
            finally:
                if hasattr(controller, "close"):
                    controller.close()
        print(f"[restore] controls={restore['status']} resolution={resolution_restore['status']}; "
              f"session metadata: {metadata_path}")
        if restore["status"] != "restored":
            print("警告：相机初始设置有字段无法确认恢复，请检查 metadata.json 中的 restore 结果。",
                  file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

