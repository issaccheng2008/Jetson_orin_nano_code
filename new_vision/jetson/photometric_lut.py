"""Offline tools for fitting and applying gray-board photometric LUTs.

The mapping is measured from corresponding physical gray-board cells.  It is
not a color-card calibration and is deliberately separate from production
vision code.  The core uses NumPy and the standard library; OpenCV is optional
for JPEG/video I/O and GUI preview.
"""
from __future__ import annotations

import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_CELL_PX = 80
DEFAULT_CENTER_FRACTION = 0.50


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def read_png(path) -> np.ndarray:
    """Read non-interlaced 8-bit grayscale/RGB/RGBA PNGs without OpenCV."""
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")
    pos, compressed = len(PNG_SIGNATURE), []
    width = height = bit_depth = color_type = interlace = None
    while pos < len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk)
            if comp or filt:
                raise ValueError("unsupported PNG compression/filter method")
        elif kind == b"IDAT":
            compressed.append(chunk)
        elif kind == b"IEND":
            break
    if (not width or not height or bit_depth != 8 or interlace != 0
            or color_type not in (0, 2, 4, 6)):
        raise ValueError("PNG must be non-interlaced 8-bit gray, RGB, or RGBA")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(b"".join(compressed))
    stride, bpp = width * channels, channels
    rows = np.empty((height, stride), dtype=np.uint8)
    offset = 0
    for y in range(height):
        filt = raw[offset]
        src = np.frombuffer(raw, np.uint8, count=stride, offset=offset + 1).astype(np.int16)
        offset += stride + 1
        dst = np.empty(stride, dtype=np.int16)
        up = rows[y - 1].astype(np.int16) if y else np.zeros(stride, np.int16)
        for x in range(stride):
            left = int(dst[x - bpp]) if x >= bpp else 0
            above = int(up[x])
            upper_left = int(up[x - bpp]) if x >= bpp else 0
            if filt == 0:
                predictor = 0
            elif filt == 1:
                predictor = left
            elif filt == 2:
                predictor = above
            elif filt == 3:
                predictor = (left + above) // 2
            elif filt == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG row filter {filt}")
            dst[x] = (int(src[x]) + predictor) & 255
        rows[y] = dst.astype(np.uint8)
    image = rows.reshape(height, width, channels) if channels > 1 else rows.reshape(height, width)
    if color_type == 2:  # PNG RGB -> OpenCV-compatible BGR.
        return image[:, :, ::-1].copy()
    if color_type == 6:  # PNG RGBA -> BGRA.
        return image[:, :, [2, 1, 0, 3]].copy()
    if color_type == 4:
        return image  # gray + alpha; caller may preserve the extra plane.
    return image


def write_png(path, image: np.ndarray) -> None:
    """Write a uint8 gray/BGR/BGRA image as a true PNG, with no overlays."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("PNG output needs a 2-D or 1/3/4-channel uint8 image")
    if image.ndim == 2:
        color_type, encoded = 0, image
    elif image.shape[2] == 1:
        color_type, encoded = 0, image[:, :, 0]
    elif image.shape[2] == 3:
        color_type, encoded = 2, image[:, :, ::-1]
    elif image.shape[2] == 4:
        color_type, encoded = 6, image[:, :, [2, 1, 0, 3]]
    else:
        raise ValueError("PNG output supports gray, BGR, or BGRA images")
    height, width = encoded.shape[:2]
    channels = 1 if color_type == 0 else 3 if color_type == 2 else 4
    raw = b"".join(b"\0" + np.ascontiguousarray(encoded[y]).tobytes()
                   for y in range(height))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff))

    payload = (PNG_SIGNATURE
               + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8,
                                                color_type, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 3))
               + chunk(b"IEND", b""))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def read_image(path) -> np.ndarray:
    """Read with OpenCV when available, otherwise accept this tool's PNGs."""
    try:
        import cv2
    except ImportError:
        return read_png(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def write_image(path, image: np.ndarray) -> None:
    if str(path).lower().endswith(".png"):
        write_png(path, image)
        return
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("non-PNG image output requires OpenCV") from exc
    if not cv2.imwrite(str(path), np.asarray(image)):
        raise OSError(f"failed to write image: {path}")


def gray_from_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("expected gray or BGR image")
    bgr = image[:, :, :3].astype(np.float32)
    return np.clip(np.rint(0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1]
                           + 0.299 * bgr[:, :, 2]), 0, 255).astype(np.uint8)


def _pava(values, weights=None, increasing=True):
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones(len(values), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if not increasing:
        values = -values
    starts, ends, means, sums = [], [], [], []
    for i, (value, weight) in enumerate(zip(values, weights)):
        starts.append(i)
        ends.append(i + 1)
        means.append(float(value))
        sums.append(float(weight))
        while len(means) >= 2 and means[-2] > means[-1]:
            total = sums[-2] + sums[-1]
            mean = (means[-2] * sums[-2] + means[-1] * sums[-1]) / total
            ends[-2] = ends[-1]
            means[-2] = mean
            sums[-2] = total
            starts.pop(); ends.pop(); means.pop(); sums.pop()
    out = np.empty(len(values), dtype=np.float64)
    for start, end, mean in zip(starts, ends, means):
        out[start:end] = mean
    return out if increasing else -out


def _group_nearby_inputs(x, y, merge_tolerance):
    order = np.argsort(x, kind="stable")
    groups = []
    for index in order:
        if not groups or float(x[index]) - float(np.median(x[groups[-1]])) > merge_tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [(float(np.median(x[g])), float(np.median(y[g])),
             float(np.ptp(y[g])) if len(g) > 1 else 0.0, g)
            for g in groups]


def fit_monotonic_lut(x, y, *, max_slope=4.0, merge_tolerance=2.5,
                      conflict_limit=16.0, outlier_floor=7.0):
    """Fit a robust monotone uint8 LUT and report conflict/outlier/extrapolation.

    Nearby duplicate input levels are first combined by median so each physical
    gray level contributes equally.  Isotonic regression suppresses local
    reversals; a residual pass rejects isolated bad points.  Slopes are capped
    before interpolation and conservative end extrapolation.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 255) & (y >= 0) & (y <= 255)
    x, y = x[valid], y[valid]
    if not len(x):
        raise ValueError("no valid fixed-to-auto gray samples")
    if not math.isfinite(max_slope) or max_slope <= 0:
        raise ValueError("max_slope must be positive and finite")
    groups = _group_nearby_inputs(x, y, merge_tolerance)
    gx = np.asarray([g[0] for g in groups], dtype=np.float64)
    gy = np.asarray([g[1] for g in groups], dtype=np.float64)
    conflicts = [{"fixed": gx[i], "auto_min": min(y[g[3]]),
                  "auto_max": max(y[g[3]]), "spread": g[2], "count": len(g[3])}
                 for i, g in enumerate(groups) if g[2] > conflict_limit]
    iso = _pava(gy)
    residual = gy - iso
    med_residual = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med_residual)))
    limit = max(float(outlier_floor), 3.0 * 1.4826 * mad)
    keep = np.abs(residual - med_residual) <= limit
    if np.count_nonzero(keep) < min(3, len(gx)):
        keep[:] = True
    outliers = [{"fixed": float(gx[i]), "auto": float(gy[i]),
                 "isotonic": float(iso[i]), "residual": float(residual[i])}
                for i in np.flatnonzero(~keep)]
    fx, fy = gx[keep], gy[keep]
    isotonic = _pava(fy)
    capped = isotonic.copy()
    for i in range(1, len(capped)):
        capped[i] = min(capped[i], capped[i - 1] + max_slope * (fx[i] - fx[i - 1]))
        capped[i] = max(capped[i], capped[i - 1])
    capped = np.clip(capped, 0.0, 255.0)
    if len(fx) == 1:
        lut_float = np.full(256, capped[0], dtype=np.float64)
        extrap_slope_low = extrap_slope_high = 0.0
        extrapolated = np.ones(256, bool)
    else:
        slopes = np.diff(capped) / np.maximum(np.diff(fx), 1e-9)
        low_slope = min(float(max_slope), max(0.0, float(slopes[0])))
        high_slope = min(float(max_slope), max(0.0, float(slopes[-1])))
        domain = np.arange(256, dtype=np.float64)
        lut_float = np.interp(domain, fx, capped)
        low = domain < fx[0]
        high = domain > fx[-1]
        lut_float[low] = np.clip(capped[0] + low_slope * (domain[low] - fx[0]), 0, capped[0])
        lut_float[high] = np.clip(capped[-1] + high_slope * (domain[high] - fx[-1]), capped[-1], 255)
        extrapolated = low | high
        extrap_slope_low, extrap_slope_high = low_slope, high_slope
    lut = np.clip(np.rint(lut_float), 0, 255).astype(np.uint8)
    # Integer rounding preserves monotonicity, but assert the contract here.
    if np.any(np.diff(lut.astype(np.int16)) < 0):
        raise AssertionError("monotone LUT construction failed")
    return lut, {
        "sample_count": int(len(x)), "unique_input_groups": int(len(groups)),
        "used_groups": int(np.count_nonzero(keep)), "max_slope": float(max_slope),
        "merge_tolerance": float(merge_tolerance), "conflict_limit": float(conflict_limit),
        "conflicts": conflicts, "outliers": outliers,
        "sample_points": [{"fixed": float(fx[i]), "auto": float(fy[i]),
                            "isotonic": float(isotonic[i]), "fit": float(capped[i])}
                           for i in range(len(fx))],
        "extrapolated_input_ranges": ([0, int(math.floor(fx[0]))] if fx[0] > 0 else None,
                                      [int(math.ceil(fx[-1])), 255] if fx[-1] < 255 else None),
        "extrapolation_slopes": {"low": extrap_slope_low, "high": extrap_slope_high},
        "lut_clipped_fraction": float(np.mean((lut == 0) | (lut == 255))),
    }


def _get_luts(mapping):
    if isinstance(mapping, (str, os.PathLike)):
        mapping = load_mapping(mapping)
    if isinstance(mapping, dict):
        bgr, gray = mapping.get("lut_bgr"), mapping.get("lut_gray")
    else:
        bgr, gray = getattr(mapping, "lut_bgr", None), getattr(mapping, "lut_gray", None)
    if bgr is None or gray is None:
        raise ValueError("mapping must provide lut_bgr and lut_gray")
    bgr, gray = np.asarray(bgr), np.asarray(gray)
    if bgr.shape != (3, 256) or bgr.dtype != np.uint8:
        raise ValueError("lut_bgr must be uint8 with shape (3, 256), ordered B,G,R")
    if gray.shape != (256,) or gray.dtype != np.uint8:
        raise ValueError("lut_gray must be uint8 with shape (256,)")
    return bgr, gray


def apply_mapping(image, mapping) -> np.ndarray:
    """Apply one path only: BGR channels for color images, gray LUT for gray.

    The input shape and channel count are preserved. BGRA's alpha plane is
    copied unchanged. Inputs must already be uint8; this helper never rescales
    or estimates a curve from the supplied image.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size:
        raise ValueError("apply_mapping expects a nonempty uint8 gray/BGR/BGRA image")
    bgr_lut, gray_lut = _get_luts(mapping)
    if image.ndim == 2:
        return gray_lut[image]
    if image.shape[2] == 1:
        return gray_lut[image[:, :, 0]][:, :, None]
    if image.shape[2] not in (3, 4):
        raise ValueError("apply_mapping supports 1, 3, or 4 channels")
    result = image.copy()
    for channel in range(3):
        result[:, :, channel] = bgr_lut[channel, image[:, :, channel]]
    return result


def load_mapping(directory) -> dict:
    directory = Path(directory)
    result = {"lut_bgr": np.load(directory / "lut_bgr.npy", allow_pickle=False),
              "lut_gray": np.load(directory / "lut_gray.npy", allow_pickle=False),
              "metadata": json.loads((directory / "mapping.json").read_text(encoding="utf-8"))}
    if (directory / "lut_max.npy").is_file():
        result["lut_max"] = np.load(directory / "lut_max.npy", allow_pickle=False)
    return result


def save_mapping(directory, lut_bgr, lut_gray, metadata, lut_max=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lut_bgr = np.asarray(lut_bgr)
    lut_gray = np.asarray(lut_gray)
    if lut_bgr.shape != (3, 256) or lut_bgr.dtype != np.uint8:
        raise ValueError("lut_bgr must be uint8 with shape (3, 256)")
    if lut_gray.shape != (256,) or lut_gray.dtype != np.uint8:
        raise ValueError("lut_gray must be uint8 with shape (256,)")
    np.save(directory / "lut_bgr.npy", lut_bgr, allow_pickle=False)
    np.save(directory / "lut_gray.npy", lut_gray, allow_pickle=False)
    if lut_max is not None:
        lut_max = np.asarray(lut_max)
        if lut_max.shape != (256,) or lut_max.dtype != np.uint8:
            raise ValueError("lut_max must be uint8 with shape (256,)")
        np.save(directory / "lut_max.npy", lut_max, allow_pickle=False)
    (directory / "mapping.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def homography_from_corners(corners, width, height):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("corners must be four finite (x,y) pairs: TL,TR,BR,BL")
    if width < 2 or height < 2:
        raise ValueError("destination width and height must be at least 2")
    dst = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1],
                      [0, height - 1]], dtype=np.float64)
    a, b = [], []
    for (x, y), (u, v) in zip(corners, dst):
        a.extend(([x, y, 1, 0, 0, 0, -u*x, -u*y],
                  [0, 0, 0, x, y, 1, -v*x, -v*y]))
        b.extend((u, v))
    coeff = np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return np.append(coeff, 1.0).reshape(3, 3)


def transform_points(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    denom = hom[:, 2]
    if np.any(np.abs(denom) < 1e-10):
        raise ValueError("projective transform sends a point to infinity")
    return hom[:, :2] / denom[:, None]


def warp_board(image, corners, cells_per_row=16, cell_px=DEFAULT_CELL_PX):
    """Perspective-rectify the board to 2 x N square cells using bilinear sampling."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("board image must be a uint8 gray/BGR image")
    if cells_per_row < 2 or cell_px < 8:
        raise ValueError("board needs >=2 cells per row and cell_px >=8")
    height, width = image.shape[:2]
    corners = np.asarray(corners, np.float64)
    if corners.shape != (4, 2) or np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= height):
        raise ValueError("board corner lies outside image bounds")
    out_w, out_h = cells_per_row * cell_px, 2 * cell_px
    h_src_to_dst = homography_from_corners(corners, out_w, out_h)
    h_dst_to_src = np.linalg.inv(h_src_to_dst)
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    src = transform_points(h_dst_to_src, np.column_stack((xx.ravel(), yy.ravel())))
    sx, sy = src[:, 0], src[:, 1]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    valid = (sx >= 0) & (sy >= 0) & (x1 < width) & (y1 < height)
    x0, x1 = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0, y1 = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    wx, wy = (sx - x0)[:, None], (sy - y0)[:, None]
    src2 = image.reshape(height, width, -1).astype(np.float32)
    p00, p10 = src2[y0, x0], src2[y0, x1]
    p01, p11 = src2[y1, x0], src2[y1, x1]
    out = ((1-wx)*(1-wy)*p00 + wx*(1-wy)*p10 + (1-wx)*wy*p01 + wx*wy*p11)
    out[~valid] = 0
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(out_h, out_w, -1)
    if image.ndim == 2:
        return out[:, :, 0], h_src_to_dst
    return out.reshape((out_h, out_w) + image.shape[2:]), h_src_to_dst


def extract_board_cells(image, corners, cells_per_row=16, cell_px=DEFAULT_CELL_PX,
                        center_fraction=DEFAULT_CENTER_FRACTION):
    """Return cell medians, stability metrics, saturation and the rectified board."""
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0,1]")
    board, src_to_board = warp_board(image, corners, cells_per_row, cell_px)
    if board.ndim == 2:
        bgr = np.repeat(board[:, :, None], 3, axis=2)
    elif board.shape[2] >= 3:
        bgr = board[:, :, :3]
    else:
        bgr = np.repeat(board[:, :, :1], 3, axis=2)
    y0_fraction, y1_fraction = (1-center_fraction)/2, (1+center_fraction)/2
    offset = int(round(cell_px * y0_fraction))
    extent = max(1, int(round(cell_px * center_fraction)))
    cells = []
    for row in range(2):
        y0 = row * cell_px + offset
        y1 = min((row + 1) * cell_px - offset, y0 + extent)
        for col in range(cells_per_row):
            x0 = col * cell_px + offset
            x1 = min((col + 1) * cell_px - offset, x0 + extent)
            patch = bgr[y0:y1, x0:x1]
            gray_patch = gray_from_bgr(patch)
            max_patch = np.max(patch, axis=2)
            level = col if row == 0 else cells_per_row - 1 - col
            cells.append({
                "row": row, "col": col, "gray_level_index": level,
                "bgr_median": np.median(patch.reshape(-1, 3), axis=0).astype(float).tolist(),
                "gray_median": float(np.median(gray_patch)),
                "max_channel_median": float(np.median(max_patch)),
                "bgr_std": np.std(patch.reshape(-1, 3), axis=0).astype(float).tolist(),
                "gray_std": float(np.std(gray_patch)),
                "max_channel_std": float(np.std(max_patch)),
                "saturated_fraction": float(np.mean(np.any((patch <= 1) | (patch >= 254), axis=2))),
                "sample_rect_board": [int(x0), int(y0), int(x1), int(y1)],
            })
    return cells, board, src_to_board


def estimate_translation(reference, image, corners, max_shift=20):
    """Estimate registration shift from board-area gradient correlation.

    This is a movement check, not an image correction. A large shift/weak match
    requires new per-photo corners so samples remain tied to physical cells.
    """
    reference, image = np.asarray(reference), np.asarray(image)
    if reference.shape[:2] != image.shape[:2]:
        return {"dx_px": None, "dy_px": None, "correlation": 0.0,
                "requires_reselection": True, "reason": "image_size_mismatch"}
    h, w = reference.shape[:2]
    c = np.asarray(corners, dtype=float)
    left, right = float(np.min(c[:, 0])), float(np.max(c[:, 0]))
    top, bottom = float(np.min(c[:, 1])), float(np.max(c[:, 1]))
    mx, my = max(6, int((right-left)*0.08)), max(6, int((bottom-top)*0.18))
    x0, x1 = max(1, int(left)-mx), min(w-1, int(right)+mx+1)
    y0, y1 = max(1, int(top)-my), min(h-1, int(bottom)+my+1)
    a = gray_from_bgr(reference)[y0:y1, x0:x1].astype(np.float64)
    b = gray_from_bgr(image)[y0:y1, x0:x1].astype(np.float64)

    def gradient(src):
        gy, gx = np.gradient(src)
        return np.hypot(gx, gy)

    a, b = gradient(a), gradient(b)

    def search(aa, bb, radius, center=(0,0), bound=None):
        best = (-2.0, center[0], center[1])
        cx, cy = center
        bound = radius if bound is None else bound
        for dy in range(max(-bound, cy-radius), min(bound, cy+radius)+1):
            ay0, by0 = max(0, dy), max(0, -dy)
            ah = min(aa.shape[0]-ay0, bb.shape[0]-by0)
            for dx in range(max(-bound, cx-radius), min(bound, cx+radius)+1):
                ax0, bx0 = max(0, dx), max(0, -dx)
                aw = min(aa.shape[1]-ax0, bb.shape[1]-bx0)
                if ah < 8 or aw < 8:
                    continue
                av = aa[ay0:ay0+ah, ax0:ax0+aw].ravel()
                bv = bb[by0:by0+ah, bx0:bx0+aw].ravel()
                av, bv = av-av.mean(), bv-bv.mean()
                den = float(np.sqrt(np.sum(av*av) * np.sum(bv*bv)))
                score = float(np.sum(av*bv) / den) if den > 1e-9 else -1.0
                if score > best[0]:
                    best = (score, dx, dy)
        return best

    # Coarse-to-fine search keeps registration inexpensive for 1280x720 frames.
    scale = max(1, int(math.ceil(max(a.shape) / 180)))
    ac, bc = a[::scale, ::scale], b[::scale, ::scale]
    coarse_limit = max(1, min(int(math.ceil(max_shift/scale)),
                              (ac.shape[1]-8)//3, (ac.shape[0]-8)//3))
    coarse = search(ac, bc, coarse_limit)
    center = (int(round(coarse[1]*scale)), int(round(coarse[2]*scale)))
    fine_radius = max(1, scale)
    score, dx, dy = search(a, b, fine_radius, center, int(max_shift))
    moved = abs(dx) > 1.5 or abs(dy) > 1.5 or score < 0.35
    return {"dx_px": int(dx), "dy_px": int(dy), "correlation": float(score),
            "requires_reselection": bool(moved),
            "reason": "board_shift_or_weak_registration" if moved else None}


def aggregate_group_cells(per_image_cells):
    """Aggregate identical physical cell positions over frames, never by area."""
    if not per_image_cells:
        raise ValueError("at least one image of a group is required")
    by_key = {}
    for image_cells in per_image_cells:
        for cell in image_cells:
            by_key.setdefault((cell["row"], cell["col"]), []).append(cell)
    result = []
    for (row, col), samples in sorted(by_key.items()):
        bgr = np.asarray([s["bgr_median"] for s in samples], dtype=float)
        gray = np.asarray([s["gray_median"] for s in samples], dtype=float)
        maximum = np.asarray([s.get("max_channel_median", s["gray_median"])
                              for s in samples], dtype=float)
        result.append({
            "row": row, "col": col, "gray_level_index": samples[0]["gray_level_index"],
            "frame_count": len(samples),
            "bgr_median": np.median(bgr, axis=0).tolist(),
            "gray_median": float(np.median(gray)),
            "max_channel_median": float(np.median(maximum)),
            "bgr_group_spread_p90_p10": (np.percentile(bgr, 90, axis=0)-np.percentile(bgr, 10, axis=0)).tolist(),
            "gray_group_spread_p90_p10": float(np.percentile(gray, 90)-np.percentile(gray, 10)),
            "max_channel_group_spread_p90_p10": float(np.percentile(maximum, 90)-np.percentile(maximum, 10)),
            "bgr_patch_std_median": np.median(np.asarray([s["bgr_std"] for s in samples],float),axis=0).tolist(),
            "gray_patch_std_median": float(np.median([s["gray_std"] for s in samples])),
            "max_channel_patch_std_median": float(np.median([s.get("max_channel_std", s["gray_std"])
                                                               for s in samples])),
            "saturated_fraction": float(np.mean([s["saturated_fraction"] for s in samples])),
            "per_frame": [{"bgr_median": s["bgr_median"], "gray_median": s["gray_median"],
                           "max_channel_median": s.get("max_channel_median", s["gray_median"]),
                           "bgr_std": s["bgr_std"], "gray_std": s["gray_std"],
                           "max_channel_std": s.get("max_channel_std", s["gray_std"]),
                           "saturated_fraction": s["saturated_fraction"]} for s in samples],
        })
    return result


def draw_line(image, p0, p1, color, thickness=1):
    """Small NumPy-only integer line renderer used for sampling overlays."""
    image = np.asarray(image)
    x0, y0 = map(int, np.rint(p0)); x1, y1 = map(int, np.rint(p1))
    steps = max(abs(x1-x0), abs(y1-y0), 1)
    xs = np.rint(np.linspace(x0, x1, steps+1)).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps+1)).astype(int)
    radius = max(0, int(thickness)//2)
    for x, y in zip(xs, ys):
        ya, yb = max(0, y-radius), min(image.shape[0], y+radius+1)
        xa, xb = max(0, x-radius), min(image.shape[1], x+radius+1)
        if ya < yb and xa < xb:
            image[ya:yb, xa:xb] = color


def draw_rect(image, x0, y0, x1, y1, color, thickness=2):
    draw_line(image, (x0,y0), (x1,y0), color, thickness)
    draw_line(image, (x1,y0), (x1,y1), color, thickness)
    draw_line(image, (x1,y1), (x0,y1), color, thickness)
    draw_line(image, (x0,y1), (x0,y0), color, thickness)

