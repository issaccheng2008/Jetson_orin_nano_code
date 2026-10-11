#!/usr/bin/env python3
"""Fit a fixed-camera-image -> auto-reference LUT from a captured gray-board session.

The two LUTs are alternatives for BGR and single-channel gray callers. This
script only exports files; it does not change or install a production mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from photometric_lut import (DEFAULT_CELL_PX, aggregate_group_cells, apply_mapping,
                             draw_line, estimate_translation,
                             extract_board_cells, fit_monotonic_lut, gray_from_bgr,
                             read_image, save_mapping, transform_points, write_png)


class FitError(RuntimeError):
    pass


def _bgr(image):
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image[:, :, :3].copy()


def _new_output_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        return path
    raise FitError(f"output directory already exists; refusing to overwrite: {path}")


def _load_corner_spec(path):
    if path is None:
        return {"auto_corners": None, "fixed_corners": None, "by_file": {}}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"auto_corners": data, "fixed_corners": None, "by_file": {}}
    if not isinstance(data, dict):
        raise FitError("corner JSON must be a four-point array or an object")
    return {"auto_corners": data.get("auto_corners", data.get("corners")),
            "fixed_corners": data.get("fixed_corners"),
            "by_file": data.get("by_file", {})}


def select_corners(image, title):
    """Interactive ordered corner selection (TL, TR, BR, BL), with original coords."""
    try:
        import cv2
    except ImportError as exc:
        raise FitError("GUI corner selection requires OpenCV; provide --corners-json") from exc
    image = _bgr(image)
    h, w = image.shape[:2]
    scale = min(1.0, 1440.0 / w, 900.0 / h)
    shown_size = (int(round(w*scale)), int(round(h*scale)))
    shown = cv2.resize(image, shown_size) if scale != 1.0 else image.copy()
    window = "Gray board corners: TL, TR, BR, BL; Enter=accept, r=reset, Esc=cancel"
    points = []

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x/scale, y/scale])

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        while True:
            canvas = shown.copy()
            for i, (x, y) in enumerate(points):
                sx, sy = int(round(x*scale)), int(round(y*scale))
                cv2.circle(canvas, (sx, sy), 6, (0, 255, 255), -1)
                cv2.putText(canvas, str(i+1), (sx+7, sy-7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xff
            if key in (10, 13) and len(points) == 4:
                return np.asarray(points, dtype=float)
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27 or key in (ord("q"), ord("Q")):
                raise FitError("corner selection cancelled")
    except cv2.error as exc:
        raise FitError(f"could not open corner-selection window: {exc}; use --corners-json") from exc
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass


def _valid_corners(corners, image, label):
    arr = np.asarray(corners, dtype=float)
    h, w = image.shape[:2]
    if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
        raise FitError(f"{label}: expected four finite (x,y) points in TL,TR,BR,BL order")
    if np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= w) or np.any(arr[:, 1] < 0) or np.any(arr[:, 1] >= h):
        raise FitError(f"{label}: corner outside the {w}x{h} image")
    edges = np.roll(arr,-1,axis=0)-arr
    next_edges = np.roll(edges,-1,axis=0)
    crosses = edges[:,0]*next_edges[:,1]-edges[:,1]*next_edges[:,0]
    if np.any(np.abs(crosses) < 1e-3) or not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise FitError(f"{label}: corners must follow a convex TL,TR,BR,BL perimeter")
    area = abs(float(np.dot(arr[:,0],np.roll(arr[:,1],-1))-
                       np.dot(arr[:,1],np.roll(arr[:,0],-1))))*.5
    if area < 25:
        raise FitError(f"{label}: selected board area is too small")
    return arr


def _corner_for_file(spec, rel_name, default):
    value = spec["by_file"].get(rel_name)
    if value is None:
        value = spec["by_file"].get(Path(rel_name).name)
    return default if value is None else value


def _choose_moved_corners(image, label, rel_name, spec, default, *, gui):
    explicit = _corner_for_file(spec, rel_name, None)
    if explicit is not None:
        return _valid_corners(explicit, image, rel_name), "coordinates_json"
    if not gui:
        raise FitError(f"board movement/weak registration at {rel_name}; add its four corners under "
                       f"by_file in --corners-json or rerun with GUI selection")
    return _valid_corners(select_corners(image, label), image, rel_name), "interactive_reselection"


def select_session_corners(session, images, spec, *, gui=True, max_shift=20):
    auto_paths, fixed_paths = images["auto"], images["fixed"]
    auto_ref = read_image(auto_paths[0])
    corners = {}
    ref_auto = spec["auto_corners"] or _corner_for_file(spec, "auto/000.png", None)
    if ref_auto is None:
        if not gui:
            raise FitError("no auto-reference corners; supply --corners-json in headless mode")
        ref_auto = select_corners(auto_ref, f"Select the gray board on {auto_paths[0].name}")
        auto_source = "interactive"
    else:
        auto_source = "coordinates_json"
    ref_auto = _valid_corners(ref_auto, auto_ref, "auto reference")
    corners["auto/000.png"] = ref_auto

    # A moved board changes the physical cells under the same pixel polygon.
    # Detect it via gradients; collect fresh corners instead of silently pairing
    # different cells. Registration is a check only, never a geometric warp of
    # the original calibration measurements.
    for group in ("auto", "fixed"):
        paths = images[group]
        group_ref = read_image(paths[0])
        group_ref_corners = (ref_auto if group == "auto" else
                             spec["fixed_corners"] or _corner_for_file(spec,"fixed/000.png",None))
        ref_source = auto_source if group == "auto" else "same_position_assumed"
        if group == "fixed" and group_ref_corners is not None:
            group_ref_corners = _valid_corners(group_ref_corners, group_ref, "fixed reference")
            ref_source = "coordinates_json"
        elif group == "fixed":
            check = estimate_translation(auto_ref, group_ref, ref_auto, max_shift=max_shift)
            if check["requires_reselection"]:
                group_ref_corners, ref_source = _choose_moved_corners(
                    group_ref, "Board moved between auto and fixed groups; select fixed-board corners",
                    "fixed/000.png", spec, None, gui=gui)
            else:
                group_ref_corners = ref_auto.copy()
        corners[f"{group}/000.png"] = group_ref_corners
        for path in paths[1:]:
            image = read_image(path)
            rel = f"{group}/{path.name}"
            explicit = _corner_for_file(spec, rel, None)
            if explicit is not None:
                corners[rel] = _valid_corners(explicit, image, rel)
                continue
            registration = estimate_translation(group_ref, image, group_ref_corners,
                                                max_shift=max_shift)
            if registration["requires_reselection"]:
                corners[rel], source = _choose_moved_corners(
                    image, f"Board moved; select corners for {rel}", rel, spec,
                    group_ref_corners, gui=gui)
            else:
                corners[rel] = group_ref_corners.copy()
    return corners


def _sample_all(paths, corner_map, cells_per_row, cell_px, center_fraction):
    per_image = []
    sample_visual = None
    for path in paths:
        image = read_image(path)
        key = f"{path.parent.name}/{path.name}"
        cells, board, h_src_to_board = extract_board_cells(
            image, corner_map[key], cells_per_row, cell_px, center_fraction)
        per_image.append(cells)
        if path.name == "000.png":
            sample_visual = (image, cells, h_src_to_board)
    return aggregate_group_cells(per_image), per_image, sample_visual


def _cell_lookup(cells):
    return {(c["row"], c["col"]): c for c in cells}


def _cell_rejection(fixed, auto, saturation_limit, spread_limit):
    reasons = []
    if fixed["saturated_fraction"] > saturation_limit:
        reasons.append("fixed_saturated")
    if auto["saturated_fraction"] > saturation_limit:
        reasons.append("auto_saturated")
    if fixed["gray_group_spread_p90_p10"] > spread_limit:
        reasons.append("fixed_unstable")
    if auto["gray_group_spread_p90_p10"] > spread_limit:
        reasons.append("auto_unstable")
    if max(fixed["bgr_group_spread_p90_p10"]) > spread_limit:
        reasons.append("fixed_channel_unstable")
    if max(auto["bgr_group_spread_p90_p10"]) > spread_limit:
        reasons.append("auto_channel_unstable")
    return reasons


def _make_curve_png(path, fits, size=(900, 560)):
    width, height = size
    canvas = np.full((height, width, 3), 255, np.uint8)
    left, top, right, bottom = 60, 30, width-25, height-45
    for i in range(5):
        x = left + (right-left)*i/4
        y = top + (bottom-top)*i/4
        draw_line(canvas, (x, top), (x, bottom), (225,225,225), 1)
        draw_line(canvas, (left, y), (right, y), (225,225,225), 1)
    draw_line(canvas, (left, bottom), (right, bottom), (30,30,30), 2)
    draw_line(canvas, (left, bottom), (left, top), (30,30,30), 2)
    channel_colors = {"B": (255,0,0), "G": (0,180,0), "R": (0,0,255), "gray": (40,40,40)}
    for name, entry in fits.items():
        lut = entry["lut"].astype(float)
        points = []
        for x in range(256):
            px = left + x/255*(right-left)
            py = bottom - lut[x]/255*(bottom-top)
            points.append((px,py))
        for a, b in zip(points, points[1:]):
            draw_line(canvas, a, b, channel_colors[name], 2)
        for sample in entry["details"].get("sample_points", []):
            px = left + sample["fixed"]/255*(right-left)
            py = bottom - sample["auto"]/255*(bottom-top)
            x,y=int(round(px)),int(round(py))
            if 0 <= x < width and 0 <= y < height:
                canvas[max(0,y-2):y+3,max(0,x-2):x+3] = channel_colors[name]
    write_png(path, canvas)


def _sampling_overlay(image, cells, h_src_to_board, cell_px):
    overlay = _bgr(image)
    board_to_src = np.linalg.inv(h_src_to_board)
    for cell in cells:
        x0,y0,x1,y1 = cell["sample_rect_board"]
        points = transform_points(board_to_src, [[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        color = (0, 220, 0) if cell["row"] == 0 else (255, 180, 0)
        for i in range(4):
            draw_line(overlay, points[i], points[(i+1)%4], color, 2)
    return overlay


def _preview_sampling_overlay(image):
    try:
        import cv2
        cv2.namedWindow("sampling boxes: press any key to continue",cv2.WINDOW_NORMAL)
        cv2.imshow("sampling boxes: press any key to continue",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"采样框预览窗口不可用；已保存 sampling_overlay.png 供检查：{exc}")


def _save_montage(path, images):
    bgrs = [_bgr(im) for im in images]
    sizes = {im.shape[:2] for im in bgrs}
    if len(sizes) != 1:
        raise FitError("comparison images have different dimensions; select matched frames/corners")
    write_png(path, np.concatenate(bgrs, axis=1))


def _build_mapping(session, auto_cells, fixed_cells, fit_count, saturation_limit,
                   spread_limit, max_slope, conflict_limit):
    auto, fixed = _cell_lookup(auto_cells), _cell_lookup(fixed_cells)
    joined, rejected = [], []
    for key in sorted(set(auto) & set(fixed)):
        a, f = auto[key], fixed[key]
        reasons = _cell_rejection(f, a, saturation_limit, spread_limit)
        record = {"row": key[0], "col": key[1], "gray_level_index": f["gray_level_index"],
                  "fixed_bgr": f["bgr_median"], "auto_bgr": a["bgr_median"],
                  "fixed_gray": f["gray_median"], "auto_gray": a["gray_median"],
                  "fixed_group_spread": f["gray_group_spread_p90_p10"],
                  "auto_group_spread": a["gray_group_spread_p90_p10"],
                  "fixed_bgr_group_spread": f["bgr_group_spread_p90_p10"],
                  "auto_bgr_group_spread": a["bgr_group_spread_p90_p10"],
                  "fixed_gray_patch_std": f["gray_patch_std_median"],
                  "auto_gray_patch_std": a["gray_patch_std_median"],
                  "fixed_bgr_patch_std": f["bgr_patch_std_median"],
                  "auto_bgr_patch_std": a["bgr_patch_std_median"],
                  "fixed_saturated_fraction": f["saturated_fraction"],
                  "auto_saturated_fraction": a["saturated_fraction"],
                  "excluded_reasons": reasons, "used_for_fit": not reasons}
        joined.append(record)
        if reasons:
            rejected.append({"row": key[0], "col": key[1], "gray_level_index": f["gray_level_index"],
                             "reasons": reasons})
    accepted = [r for r in joined if r["used_for_fit"]]
    if len(accepted) < 4:
        raise FitError(f"only {len(accepted)} stable, unsaturated cells remain; need at least 4")
    bgr_fits = {}
    for channel, name in enumerate(("B", "G", "R")):
        lut, details = fit_monotonic_lut(
            [r["fixed_bgr"][channel] for r in accepted],
            [r["auto_bgr"][channel] for r in accepted],
            max_slope=max_slope, conflict_limit=conflict_limit)
        bgr_fits[name] = {"lut": lut, "details": details}
    gray_lut, gray_details = fit_monotonic_lut(
        [r["fixed_gray"] for r in accepted], [r["auto_gray"] for r in accepted],
        max_slope=max_slope, conflict_limit=conflict_limit)
    all_conflicts = {name: data["details"]["conflicts"] for name, data in bgr_fits.items()}
    all_conflicts["gray"] = gray_details["conflicts"]
    return (np.stack([bgr_fits[c]["lut"] for c in ("B","G","R")]), gray_lut,
            joined, rejected, bgr_fits, gray_details, all_conflicts)


def _holdout_report(fixed_holdout, auto_holdout, lut_bgr, lut_gray, per_image_fixed,
                    per_image_auto):
    fixed = _cell_lookup(fixed_holdout)
    auto = _cell_lookup(auto_holdout)
    by_level = {}
    errors = []
    for key in sorted(set(fixed)&set(auto)):
        f, a = fixed[key], auto[key]
        fixed_bgr = np.asarray(f["bgr_median"], dtype=float)
        auto_bgr = np.asarray(a["bgr_median"], dtype=float)
        mapped_bgr = lut_bgr[np.arange(3), np.clip(np.rint(fixed_bgr),0,255).astype(int)]
        before = fixed_bgr-auto_bgr
        after = mapped_bgr.astype(float)-auto_bgr
        fixed_g, auto_g = float(f["gray_median"]), float(a["gray_median"])
        mapped_g = float(lut_gray[int(np.clip(round(fixed_g),0,255))])
        level = f["gray_level_index"]
        entry = {"row": key[0], "col": key[1], "gray_level_index": level,
                 "fixed_bgr": fixed_bgr.tolist(), "auto_bgr": auto_bgr.tolist(),
                 "mapped_bgr": mapped_bgr.astype(float).tolist(),
                 "before_error_bgr": before.tolist(), "after_error_bgr": after.tolist(),
                 "fixed_gray": fixed_g, "auto_gray": auto_g, "mapped_gray": mapped_g,
                 "before_error_gray": fixed_g-auto_g, "after_error_gray": mapped_g-auto_g,
                 "fixed_saturated_fraction": f["saturated_fraction"],
                 "auto_saturated_fraction": a["saturated_fraction"]}
        errors.append(entry)
        by_level.setdefault(int(level), []).append(entry)
    before_gray = [abs(e["before_error_gray"]) for e in errors]
    after_gray = [abs(e["after_error_gray"]) for e in errors]
    row_repeat = []
    for level, items in sorted(by_level.items()):
        row_repeat.append({"gray_level_index": level,
                           "auto_gray_upper_lower_delta": (abs(items[0]["auto_gray"]-items[1]["auto_gray"])
                                                           if len(items) >= 2 else None),
                           "mapped_error_upper_lower_delta": (abs(items[0]["after_error_gray"]-
                                                                   items[1]["after_error_gray"])
                                                              if len(items) >= 2 else None)})
    level_summary = []
    for level in sorted(by_level):
        entries = by_level[level]
        level_summary.append({"gray_level_index": level,
                              "fixed_gray": float(np.mean([e["fixed_gray"] for e in entries])),
                              "auto_gray": float(np.mean([e["auto_gray"] for e in entries])),
                              "mapped_gray": float(np.mean([e["mapped_gray"] for e in entries])),
                              "before_abs_error": float(np.mean([abs(e["before_error_gray"]) for e in entries])),
                              "after_abs_error": float(np.mean([abs(e["after_error_gray"]) for e in entries]))})
    selected = {}
    for label, level in (("black", 0), ("middle_gray", max(by_level)//2 if by_level else 0),
                         ("white", max(by_level) if by_level else 0)):
        selected[label] = next((e for e in level_summary if e["gray_level_index"] == level), None)
    if all(selected.get(k) for k in ("black", "middle_gray", "white")):
        for kind in ("fixed_gray", "auto_gray", "mapped_gray"):
            selected.setdefault("black_to_white_gray_difference", {})[kind] = (
                selected["white"][kind]-selected["black"][kind])
    # Clipping is measured on the held-out full frames separately from board
    # sample saturation; the caller adds the full-frame measurements.
    return {"held_out_images_per_group": len(per_image_fixed),
            "gray_mae_before": float(np.mean(before_gray)) if before_gray else None,
            "gray_mae_after": float(np.mean(after_gray)) if after_gray else None,
            "gray_rmse_before": float(np.sqrt(np.mean(np.square([e["before_error_gray"] for e in errors])))) if errors else None,
            "gray_rmse_after": float(np.sqrt(np.mean(np.square([e["after_error_gray"] for e in errors])))) if errors else None,
            "block_errors": errors, "by_gray_level": level_summary,
            "black_middle_white": selected, "upper_lower_repeat_check": row_repeat}


def _training_row_repeat_check(cells, input_tolerance=2.5, target_conflict_limit=16.0):
    by_level = {}
    for cell in cells:
        by_level.setdefault(int(cell["gray_level_index"]), []).append(cell)
    report = []
    conflicts = []
    for level, pair in sorted(by_level.items()):
        if len(pair) < 2:
            continue
        upper = next((c for c in pair if c["row"] == 0), pair[0])
        lower = next((c for c in pair if c["row"] == 1), pair[-1])
        fixed_delta = abs(float(upper["fixed_gray"])-float(lower["fixed_gray"]))
        auto_delta = abs(float(upper["auto_gray"])-float(lower["auto_gray"]))
        fixed_bgr_delta = np.abs(np.asarray(upper["fixed_bgr"],float)-
                                 np.asarray(lower["fixed_bgr"],float))
        auto_bgr_delta = np.abs(np.asarray(upper["auto_bgr"],float)-
                                np.asarray(lower["auto_bgr"],float))
        same_fixed = (fixed_delta <= input_tolerance
                      and bool(np.all(fixed_bgr_delta <= input_tolerance)))
        target_conflict = (same_fixed and
                           (auto_delta > target_conflict_limit
                            or bool(np.any(auto_bgr_delta > target_conflict_limit))))
        row = {"gray_level_index": level,
               "fixed_gray_upper_lower_delta": fixed_delta,
               "auto_gray_upper_lower_delta": auto_delta,
               "fixed_bgr_upper_lower_delta": fixed_bgr_delta.tolist(),
               "auto_bgr_upper_lower_delta": auto_bgr_delta.tolist(),
               "same_fixed_response": same_fixed,
               "target_conflict": bool(target_conflict)}
        report.append(row)
        if target_conflict:
            conflicts.append(row)
    return report, conflicts


def _apply_validation(path, output_dir, mapping, mode="auto"):
    path = Path(path)
    video_suffixes = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
    if path.suffix.lower() in video_suffixes:
        try:
            import cv2
        except ImportError as exc:
            raise FitError("video validation requires OpenCV on this machine") from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise FitError(f"cannot open validation video: {path}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        out_path = output_dir / f"{path.stem}_mapped.avi"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))
        if not writer.isOpened():
            cap.release()
            raise FitError(f"cannot create validation video: {out_path}")
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if mode == "gray":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mapped = apply_mapping(gray, mapping)
                    mapped = cv2.cvtColor(mapped, cv2.COLOR_GRAY2BGR)
                else:
                    mapped = apply_mapping(frame, mapping)
                writer.write(mapped)
                count += 1
        finally:
            cap.release(); writer.release()
        return {"input": str(path), "output": str(out_path), "frames": count,
                "mode": mode, "refit": False, "fps": fps}
    image = read_image(path)
    if mode == "gray" and image.ndim == 3:
        image = gray_from_bgr(image)
    mapped = apply_mapping(image, mapping)
    out_path = output_dir / f"{path.stem}_mapped.png"
    write_png(out_path, mapped)
    return {"input": str(path), "output": str(out_path), "mode": mode,
            "refit": False, "input_shape": list(image.shape)}


def fit_session(args):
    session = Path(args.session).resolve()
    if not session.is_dir():
        raise FitError(f"session directory does not exist: {session}")
    metadata_path = session / "metadata.json"
    if not metadata_path.is_file():
        raise FitError("metadata.json is missing; expected output from capture_photometric_pairs.py")
    capture_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    capture_warnings = {}
    for group in ("auto", "fixed"):
        result = capture_metadata.get("groups", {}).get(group)
        if result and (result.get("status") != "complete"
                       or not result.get("drift", {}).get("qualified", False)):
            if (not getattr(args, "allow_capture_warnings", False)
                    or result.get("status") not in ("complete", "unqualified_drift")
                    or not result.get("settings", {}).get("verified", False)
                    or len(result.get("frames", [])) != 10):
                raise FitError(f"{group} group is not qualified for fitting: "
                               f"status={result.get('status')} drift={result.get('drift')}; "
                               "use --allow-capture-warnings for a fully captured group only")
            capture_warnings[group] = {"status": result["status"],
                                       "stability": result.get("stability"),
                                       "drift": result.get("drift")}
    images = {g: sorted((session/g).glob("*.png")) for g in ("auto", "fixed")}
    if any(len(images[g]) != 10 for g in images):
        raise FitError("expected exactly ten auto/*.png and ten fixed/*.png images")
    if not 2 <= args.fit_images < 10 or args.held_out_images != 10-args.fit_images:
        raise FitError("fit/holdout split must cover ten frames; default is 8 fit + 2 holdout")
    spec = _load_corner_spec(args.corners_json)
    corner_map = select_session_corners(session, images, spec,
                                        gui=not args.no_gui, max_shift=args.max_alignment_shift_px)
    # Each pair represents the same physical locations and must have equal size.
    for i in range(10):
        if read_image(images["auto"][i]).shape != read_image(images["fixed"][i]).shape:
            raise FitError(f"paired frame dimensions differ at index {i}; recapture with matched settings")

    auto_train, _, auto_visual = _sample_all(
        images["auto"][:args.fit_images], corner_map, args.cells_per_row,
        args.cell_px, args.center_fraction)
    fixed_train, _, _ = _sample_all(
        images["fixed"][:args.fit_images], corner_map, args.cells_per_row,
        args.cell_px, args.center_fraction)
    lut_bgr, lut_gray, training_cells, rejected, bgr_fits, gray_details, conflicts = _build_mapping(
        session, auto_train, fixed_train, args.fit_images, args.max_saturation_fraction,
        args.max_patch_spread, args.max_slope, args.conflict_limit)
    mapping = {"lut_bgr": lut_bgr, "lut_gray": lut_gray}

    # Holdout stats are independently extracted from the last two pairs; they
    # never enter fitting, duplicate grouping, outlier rejection, or LUT updates.
    auto_holdout, auto_hold_frames, _ = _sample_all(
        images["auto"][args.fit_images:], corner_map, args.cells_per_row,
        args.cell_px, args.center_fraction)
    fixed_holdout, fixed_hold_frames, _ = _sample_all(
        images["fixed"][args.fit_images:], corner_map, args.cells_per_row,
        args.cell_px, args.center_fraction)
    quality = _holdout_report(fixed_holdout, auto_holdout, lut_bgr, lut_gray,
                              fixed_hold_frames, auto_hold_frames)

    output = Path(args.output_dir).resolve() if args.output_dir else session / "photometric_mapping"
    output = _new_output_dir(output)
    overlay = _sampling_overlay(auto_visual[0], auto_visual[1], auto_visual[2], args.cell_px)
    write_png(output / "sampling_overlay.png", overlay)
    if not args.no_gui:
        _preview_sampling_overlay(overlay)
    _make_curve_png(output / "mapping_curve.png",
                    {**bgr_fits, "gray": {"lut": lut_gray, "details": gray_details}})

    compare_index = min(args.comparison_index, 9)
    compare_fixed = read_image(images["fixed"][compare_index])
    compare_auto = read_image(images["auto"][compare_index])
    compare_mapped = apply_mapping(compare_fixed, mapping)
    write_png(output / f"comparison_fixed_{compare_index:03d}.png", compare_fixed)
    write_png(output / f"comparison_mapped_{compare_index:03d}.png", compare_mapped)
    write_png(output / f"comparison_auto_{compare_index:03d}.png", compare_auto)
    _save_montage(output / f"comparison_{compare_index:03d}.png",
                  [compare_fixed, compare_mapped, compare_auto])
    quality["comparison_images"] = [f"fixed original", "fixed after LUT", "auto reference"]
    quality["comparison_index"] = compare_index
    quality["capture_warnings_accepted"] = capture_warnings

    # Full-frame clipping/crop metrics use both held-out fixed frames.
    input_saturation, output_clip = [], []
    for path in images["fixed"][args.fit_images:]:
        image = read_image(path)
        if image.ndim == 3 and image.shape[2] >= 3:
            channels = image[:, :, :3]
        elif image.ndim == 2:
            channels = image[:, :, None]
        else:
            channels = image[:, :, :1]
        mapped = apply_mapping(image, mapping)
        mapped_channels = mapped[:, :, :3] if mapped.ndim == 3 and mapped.shape[2] >= 3 else mapped.reshape(mapped.shape[:2] + (1,))
        input_saturation.append(float(np.mean((channels <= 1) | (channels >= 254))) )
        output_clip.append(float(np.mean((mapped_channels == 0) | (mapped_channels == 255))))
    quality["full_frame_saturation_fraction"] = float(np.mean(input_saturation))
    quality["full_frame_lut_clipping_fraction"] = float(np.mean(output_clip))
    quality["holdout_images_are_independent_scenes"] = False
    quality["holdout_note"] = "last two frames are same-scene repeats; this is not independent-scene generalization"
    quality["training_cell_count"] = len(training_cells)
    quality["accepted_cell_count"] = sum(r["used_for_fit"] for r in training_cells)
    quality["excluded_cells"] = rejected
    quality["mapping_conflicts_by_channel"] = conflicts
    train_repeat, train_conflicts = _training_row_repeat_check(
        [r for r in training_cells if r["used_for_fit"]],
        target_conflict_limit=args.conflict_limit)
    quality["training_upper_lower_repeat_check"] = train_repeat
    quality["training_physical_cell_conflicts"] = train_conflicts
    quality["conflict_warning"] = bool(any(conflicts.values()) or train_conflicts)

    bgr_before = np.asarray([r["fixed_bgr"] for r in training_cells if r["used_for_fit"]], float)
    bgr_after = np.asarray([[lut_bgr[c, int(np.clip(round(v[c]),0,255))] for c in range(3)]
                            for v in bgr_before], float)
    bgr_target = np.asarray([r["auto_bgr"] for r in training_cells if r["used_for_fit"]], float)
    quality["training_median_abs_error_bgr_before"] = np.median(np.abs(bgr_before-bgr_target),axis=0).tolist()
    quality["training_median_abs_error_bgr_after"] = np.median(np.abs(bgr_after-bgr_target),axis=0).tolist()
    quality["lut_bgr_extrapolated_input_ranges"] = {
        c: bgr_fits[c]["details"]["extrapolated_input_ranges"] for c in ("B","G","R")}
    quality["lut_gray_extrapolated_input_ranges"] = gray_details["extrapolated_input_ranges"]

    validation_outputs = []
    if args.apply_to:
        validation_dir = output / "extra_validation"
        validation_dir.mkdir()
        for path in args.apply_to:
            validation_outputs.append(_apply_validation(path, validation_dir, mapping,
                                                       args.validation_mode))
    quality["extra_validation"] = validation_outputs

    fit_summary = {
        "schema_version": 1, "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "session": str(session), "capture_session_id": capture_metadata.get("session_id"),
        "mapping_direction": "fixed camera image -> auto-setting reference image",
        "source": {"auto_reference": "captured auto group measurements",
                   "fixed_input": "captured fixed group measurements",
                   "same_physical_cell_pairing": True},
        "board": {"cells_per_row": args.cells_per_row, "physical_rows": 2,
                  "upper_row_gray_order": "black_to_white",
                  "lower_row_gray_order": "white_to_black",
                  "center_fraction_sampled": args.center_fraction,
                  "rectified_cell_px": args.cell_px,
                  "source_rgb_print_values_used": False},
        "fit_split": {"fit_images_per_group": args.fit_images,
                      "holdout_images_per_group": args.held_out_images,
                      "holdout_indices": list(range(args.fit_images,10)),
                      "same_scene_repeatability_only": True},
        "fit_method": {"method": "median gray-cell samples + robust isotonic regression + piecewise linear LUT",
                       "max_slope_output_per_input": args.max_slope,
                       "saturated_cell_fraction_limit": args.max_saturation_fraction,
                       "unstable_cell_p90_p10_limit_dn": args.max_patch_spread,
                       "same_input_conflict_limit_dn": args.conflict_limit,
                       "polynomial_fit": False,
                       "force_black_to_zero_or_white_to_255": False},
        "selected_corners_by_file": {key: np.asarray(value,float).tolist()
                                      for key,value in corner_map.items()},
        "lut_bgr_order": ["B","G","R"], "lut_bgr_shape": list(lut_bgr.shape),
        "lut_gray_shape": list(lut_gray.shape),
        "per_channel_fit": {**{k:v["details"] for k,v in bgr_fits.items()},
                            "gray": gray_details},
        "training_cells": training_cells, "quality": quality,
        "scope_note": "Gray-board response match only; not arbitrary color recovery. BGR and gray LUT paths are alternatives, never stack them.",
        "production_deployed": False,
    }
    save_mapping(output, lut_bgr, lut_gray, fit_summary)
    (output / "quality.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                                          encoding="utf-8")
    with (output / "block_errors.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        rows = quality["block_errors"]
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle,fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    (output / "corners_used.json").write_text(json.dumps(
        {"by_file": fit_summary["selected_corners_by_file"]},ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    return output, fit_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="capture_photometric_pairs.py session directory")
    parser.add_argument("--corners-json", default=None,
                        help="four ordered auto-reference corners or JSON with auto_corners/fixed_corners/by_file")
    parser.add_argument("--no-gui", action="store_true",
                        help="require all needed corners in --corners-json")
    parser.add_argument("--allow-capture-warnings", action="store_true",
                        help="explicitly fit fully captured groups with quality warnings; "
                             "retain warnings and per-cell rejection")
    parser.add_argument("--cells-per-row", type=int, default=16)
    parser.add_argument("--fit-images", type=int, default=8,
                        help="first N paired frames used to fit (default 8)")
    parser.add_argument("--held-out-images", type=int, default=2,
                        help="remaining paired frames for same-scene repeatability (default 2)")
    parser.add_argument("--cell-px", type=int, default=DEFAULT_CELL_PX)
    parser.add_argument("--center-fraction", type=float, default=0.50,
                        help="sample only the central fraction of each block")
    parser.add_argument("--max-slope", type=float, default=4.0)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.40)
    parser.add_argument("--max-patch-spread", type=float, default=8.0,
                        help="exclude cells whose temporal 90-10 range exceeds this DN")
    parser.add_argument("--conflict-limit", type=float, default=16.0,
                        help="report conflicting targets for nearly equal fixed inputs")
    parser.add_argument("--max-alignment-shift-px", type=int, default=20)
    parser.add_argument("--comparison-index", type=int, default=8,
                        help="paired same-scene frame for comparison, default holdout frame 8")
    parser.add_argument("--output-dir", default=None,
                        help="new empty output directory; default session/photometric_mapping")
    parser.add_argument("--apply-to", nargs="*", default=[],
                        help="optional extra images/videos; apply this fixed LUT without refitting")
    parser.add_argument("--validation-mode", choices=("auto","bgr","gray"), default="auto",
                        help="extra inputs use BGR or gray LUT; auto follows input channels")
    args = parser.parse_args(argv)
    try:
        output, result = fit_session(args)
    except (FitError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"LUT 已导出到：{output}")
    print(f"LUT 单调性：通过；BGR shape=(3,256)，gray shape=(256,)")
    print(f"holdout 灰度块 MAE：{result['quality']['gray_mae_before']:.2f} -> "
          f"{result['quality']['gray_mae_after']:.2f} DN（同场景重复性，不是泛化验证）")
    if result["quality"]["conflict_warning"]:
        print("警告：同一固定灰度附近存在参考值冲突；请查看 quality.json 的 mapping_conflicts_by_channel。")
    print("三通道 LUT 只匹配灰阶板的中性色响应，不代表任意颜色的准确恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

