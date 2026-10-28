#!/usr/bin/env python3
"""Offline card stages and unfiltered lane-contrast comparison; no robot I/O.

Input manifest: list of {id, image, pitch_deg, truth, roi_xyxy}; image paths are
relative to the manifest. ROI is a manual display crop only, never detector input.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from shape_detector import ShapeDetector, WORK_W, WORK_H, _quad_4pt
from card_appearance import card_appearance
from camera_config import load as load_camera
from line_detector_v1_warp import LineDetector
from photometric_thresholds import load_lut_mapping, measure, MAX_CHANNEL_REFERENCE


def save(path, pixels):
    ok, encoded = cv2.imencode(path.suffix, pixels)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    encoded.tofile(path)


def cell(pixels, label, width=480, height=300, binary=False):
    canvas = np.full((height, width, 3), 32, np.uint8)
    if pixels is not None and pixels.size:
        if pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        scale = min(width/pixels.shape[1], (height-30)/pixels.shape[0])
        w, h = max(1, round(pixels.shape[1]*scale)), max(1, round(pixels.shape[0]*scale))
        resized = cv2.resize(pixels, (w, h), interpolation=(cv2.INTER_NEAREST if binary
                                                                          else cv2.INTER_AREA))
        x, y = (width-w)//2, 30+(height-30-h)//2
        canvas[y:y+h, x:x+w] = resized
    else:
        cv2.putText(canvas, 'NOT REACHED', (12, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'No verified card box', (12, height//2+25),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .52,
                (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def unfiltered_contrast():
    # Exact committed CPU implementation preceding the isolated thin-line change.
    source = subprocess.check_output(
        ['git', 'show', '222f72a^:new_vision/jetson/line_preprocess.py'], cwd=ROOT)
    module = types.ModuleType('line_preprocess_before_width')
    exec(compile(source, '222f72a^:line_preprocess.py', 'exec'), module.__dict__)
    return module.extract_lane_candidates, hashlib.sha256(source).hexdigest()


def crop(image, box):
    x0, y0, x1, y1 = np.asarray(box, int)
    return image[max(0, y0):min(image.shape[0], y1),
                 max(0, x0):min(image.shape[1], x1)]


def contrast_box_probe(mask, lane, detector, dbg, item, frame_shape):
    """Annotation-associated diagnostic only, NOT an automatic card decision.

    Retrieve contours without annotations; then use the manual display ROI to
    associate one contour with the actual card for measuring its failure stage.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None, []
    candidates = []
    reference = np.asarray(item['roi_xyxy'], float)
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1 or hierarchy[0][i][2] == -1:
            continue
        quad = _quad_4pt(contour)
        if quad is None:
            continue
        camera_quad = cv2.perspectiveTransform(quad[None], lane.M_inv)[0]
        lo, hi = camera_quad.min(axis=0), camera_quad.max(axis=0)
        inter = np.maximum(0, np.minimum(hi, reference[2:])-np.maximum(lo, reference[:2])).prod()
        union = (hi-lo).prod()+(reference[2:]-reference[:2]).prod()-inter
        overlap = inter/max(1., union)
        if overlap >= .25:
            candidates.append((overlap, camera_quad))
    if not candidates:
        return None, []
    _, q = max(candidates, key=lambda x:x[0])
    q *= [WORK_W/frame_shape[1], WORK_H/frame_shape[0]]
    detector._geom_rejects = {}
    accepted = detector._geom_ok(q, from_ring=True)
    app = card_appearance(dbg['gray'], q)
    local = detector._classify_contour(app.symbol) if app.symbol is not None else None
    warp = detector._warp_card(cv2.bitwise_or(dbg['binary'], dbg['ink']), q)
    old = detector._classify(warp, {})
    result = dict(association='manual ROI for diagnosis only', quad_work=q.tolist(),
                  geometry_accepted=bool(accepted), geometry_reject=dict(detector._geom_rejects),
                  appearance_reason=app.reason, local_shape=local, old_same_box_shape=old)
    return result, [cell(app.gray, 'Contrast box / gray', 280, 300),
                    cell(app.binary, 'Local: '+str(local), 280, 300, True),
                    cell(warp, 'Old: '+str(old), 280, 300, True)]


def synthetic_geometry():
    """Known ideal ground squares: test geometry gates, NOT image recognition."""
    det = ShapeDetector()
    f = WORK_H/(2*np.tan(np.radians(det.cfg['cam_vfov_deg'])/2))
    out = []
    for z in (20., 30., 40.):
        for x in (-30., -20., -10., 0., 10., 20., 30.):
            for yaw in (0., 30., 60.):
                for actual_pitch in (35., 45., 55.):
                    for roll in (0., 15.):
                        a = np.radians(yaw)
                        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                        ground = np.array([[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]]) @ rot.T
                        ground += [x, z]
                        theta = np.radians(actual_pitch)
                        depth = det.cfg['cam_height_cm']*np.sin(theta)+ground[:, 1]*np.cos(theta)
                        uv = np.column_stack((f*ground[:, 0]/depth,
                            f*(det.cfg['cam_height_cm']*np.cos(theta)-ground[:, 1]*np.sin(theta))/depth))
                        a = np.radians(roll)
                        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                        uv = (uv @ rot.T + [WORK_W/2, WORK_H/2]).astype(np.float32)
                        det._geom_rejects = {}
                        valid = det._geom_ok(uv, from_ring=True)
                        out.append(dict(x_cm=x, z_model_cm=z, yaw_deg=yaw,
                            actual_pitch_deg=actual_pitch, assumed_pitch_deg=45., roll_deg=roll,
                            all_corners_visible=bool(np.all(uv >= 0) and
                                np.all(uv[:, 0] < WORK_W) and np.all(uv[:, 1] < WORK_H)),
                            geometry_accepted=bool(valid), reject=','.join(det._geom_rejects)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lut-dir', type=Path, default=ROOT/'new_vision/config/photometric_mapping')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load_lut_mapping(args.lut_dir)
    no_width, baseline_hash = unfiltered_contrast()
    camera = load_camera()
    rows = []
    for item in json.loads(args.manifest.read_text(encoding='utf-8')):
        image_path = args.manifest.parent/item['image']
        frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(str(image_path))
        detector = ShapeDetector()
        detector.photometric_mode = 'lut'
        detector.set_camera_pitch_deg(item['pitch_deg'])
        _, dbg = detector.update(frame, classify=True, issue_action=False, proposal_mode='all')
        q = dbg.get('quad_work')
        old = detector._classify(dbg['warp'], {}) if q is not None else None
        appearance = card_appearance(dbg['gray'], q) if q is not None else None
        local = (detector._classify_contour(appearance.symbol)
                 if appearance is not None and appearance.symbol is not None else None)
        original = frame.copy()
        if q is not None:
            cv2.polylines(original, [dbg['quad'].astype(np.int32)], True, (0, 255, 0), 3)
        pipeline = [cell(original, '1 Camera / detected box'),
                    cell(dbg['gray'], '2 Gray + LUT / 960x540'),
                    cell(dbg['binary'], '3 Card selective binary', binary=True),
                    cell(dbg['ink'], '4 Ring ink / Otsu', binary=True),
                    cell(dbg.get('appearance_gray'), '5 Rectified gray / 200x200'),
                    cell(dbg.get('appearance_binary'), '6 Local binary / '+str(local), binary=True)]
        save(args.output/(item['id']+'_pipeline.jpg'),
             np.vstack((np.hstack(pipeline[:3]), np.hstack(pipeline[3:]))))
        scale = np.array([dbg['gray'].shape[1]/frame.shape[1], dbg['gray'].shape[0]/frame.shape[0]]*2)
        work_box = np.asarray(item['roi_xyxy'])*scale
        zooms = [cell(crop(frame, item['roi_xyxy']), 'Camera crop', 240, 270),
                 cell(crop(dbg['binary'], work_box), 'Selective crop', 240, 270, True),
                 cell(crop(dbg['ink'], work_box), 'Ring ink crop', 240, 270, True),
                 cell(dbg.get('appearance_gray'), 'Rectified gray', 240, 270),
                 cell(dbg.get('appearance_binary'), 'Local: '+str(local), 240, 270, True),
                 cell(dbg.get('warp'), 'Old binary: '+str(old), 240, 270, True)]
        save(args.output/(item['id']+'_card_zoom.png'), np.hstack(zooms))

        # Match the lane's static camera angle, MAX-channel luminance and LUT.
        # This is a visualization comparison only; never fed to card voting.
        lane = LineDetector(cam_w=frame.shape[1], cam_h=frame.shape[0],
            cam_height_cm=camera['mount_height_cm'], cam_pitch_deg=camera['pitch_deg'],
            cam_vfov_deg=camera['vfov_deg'])
        bird = cv2.warpPerspective(frame, lane.M, (lane.bird_w, lane.bird_h))
        bird[~lane.ground_valid_mask] = 0
        photo = measure(frame.max(axis=2), 'lut', MAX_CHANNEL_REFERENCE, lut_input='max_channel')
        gray_bird = photo.apply(bird.max(axis=2))
        binary_bird = no_width(gray_bird, 'contrast', photometry=photo,
                              valid_mask=lane.ground_valid_mask)[1]
        save(args.output/(item['id']+'_bird.png'), bird)
        save(args.output/(item['id']+'_contrast_no_width.png'), binary_bird)
        save(args.output/(item['id']+'_bird_compare.jpg'), np.hstack([
            cell(bird, 'Metric IPM / static 45 deg', 550, 420),
            cell(binary_bird, 'Contrast BEFORE thin-line filter', 550, 420, True)]))
        x0,y0,x1,y1=item['roi_xyxy']
        corners=np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]])
        dst=cv2.perspectiveTransform(corners, lane.M)[0]
        box=[*np.floor(dst.min(axis=0)).astype(int), *np.ceil(dst.max(axis=0)).astype(int)]
        save(args.output/(item['id']+'_contrast_card_zoom.png'), np.hstack([
            cell(crop(bird,box), 'Metric IPM card crop', 330, 300),
            cell(crop(binary_bird,box), 'Contrast / no width filter', 330, 300, True),
            cell(dbg.get('appearance_binary'), 'Current card local binary', 330, 300, True)]))
        probe, probe_panels = contrast_box_probe(binary_bird, lane, detector, dbg, item, frame.shape)
        if probe_panels:
            save(args.output/(item['id']+'_contrast_box_probe.png'), np.hstack(probe_panels))
        row = dict(id=item['id'], truth=item['truth'], pitch_deg=item['pitch_deg'],
                   recorded_shape=item.get('recorded_shape'), current_shape=dbg.get('shape'),
                   old_same_box_shape=old, local_same_box_shape=local,
                   verified_box=bool(dbg.get('card_found')),
                   failure_stage=('classified' if dbg.get('shape') else
                                  'classification' if q is not None else 'box_proposal_or_geometry'),
                   proposal_counts=dbg.get('shape_proposal_counts'), quad_total=dbg['quad_total'],
                   quad_geom=dbg['quad_geom'], geom_rejects=dbg['geom_rejects'],
                   appearance_reason=appearance.reason if appearance else None,
                   classification_path=dbg.get('shape_classification_path'),
                   source_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                   bird_size=[lane.bird_w,lane.bird_h], contrast_box_probe=probe)
        rows.append(row)
        print(item['id'], row['current_shape'], row['failure_stage'], flush=True)
    audit=dict(opencv_version=cv2.__version__, images=rows,
        baseline_contrast_sha256=baseline_hash, synthetic_scope=
        'Ideal pinhole ground quads, no image detection, no motor/action execution',
        source_sha256={p:hashlib.sha256((ROOT/'new_vision/jetson'/p).read_bytes()).hexdigest()
            for p in ['shape_detector.py','card_appearance.py','line_detector_v1_warp.py',
                      'photometric_thresholds.py','line_preprocess.py']},
        lut_sha256={p:hashlib.sha256((args.lut_dir/p).read_bytes()).hexdigest()
                    for p in ['lut_gray.npy','lut_max.npy','mapping.json']})
    (args.output/'results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    geometry=synthetic_geometry()
    with (args.output/'synthetic_geometry.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=geometry[0]); writer.writeheader(); writer.writerows(geometry)


if __name__ == '__main__':
    main()
