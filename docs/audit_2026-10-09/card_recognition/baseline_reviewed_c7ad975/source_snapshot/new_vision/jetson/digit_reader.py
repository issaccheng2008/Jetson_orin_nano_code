"""Downward camera — dash counting + digit recognition for track positioning.

Robot-left downward camera, white floor + black dashes/numbers.
Dashes pass from right to left across the frame. Numbers appear occasionally
for absolute calibration.
"""
import cv2
import numpy as np
import os
import time


class DownwardReader:
    def __init__(self, cam_idx=1, cam_w=640, cam_h=480,
                 dash_roi=None, digit_roi=None):
        """
        dash_roi: (x, y, w, h) — vertical strip for dash crossing detection
        digit_roi: (x, y, w, h) — area where numbers may appear
        """
        self.cam_idx = cam_idx
        self.cam_w = cam_w
        self.cam_h = cam_h

        # Default ROIs (full frame, tune later)
        self.dash_roi = dash_roi or (cam_w // 2 - 30, 0, 60, cam_h)
        self.digit_roi = digit_roi or (0, 0, cam_w, cam_h)

        # ── Preprocessing params ──
        self.clahe_clip = 2.0
        self.clahe_grid = 8

        # ── Dash detection ──
        self.dash_black_ratio = 0.30      # min fraction of black pixels in ROI column
        self.dash_min_width_px = 3        # min dash width in pixels
        self.dash_debounce_frames = 3     # frames to confirm dash entry/exit
        self.dashes_per_number = None     # set after first calibration

        # ── Template matching digit recognition ──
        self._templates = {}              # digit -> list of template images
        self._build_templates()

        # ── State ──
        self._dash_count = 0
        self._last_number = None
        self._dash_inside = False
        self._dash_debounce = 0
        self._prev_black_frac = 0.0
        self._cap = None

    # ═══════════════════════════════════════════════════════════
    # Template building
    # ═══════════════════════════════════════════════════════════

    def _build_templates(self):
        """Generate clean digit templates at multiple scales."""
        fonts = [cv2.FONT_HERSHEY_SIMPLEX]
        scales = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        thicknesses = [1, 2, 3]

        for digit in range(10):
            temps = []
            ch = str(digit)
            for font in fonts:
                for scale in scales:
                    for thick in thicknesses:
                        img = np.ones((80, 60), dtype=np.uint8) * 255
                        (tw, th), _ = cv2.getTextSize(ch, font, scale, thick)
                        tx = (60 - tw) // 2
                        ty = (80 + th) // 2
                        cv2.putText(img, ch, (tx, ty), font, scale, 0, thick)
                        _, bw = cv2.threshold(img, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # Crop to content
                        ys, xs = np.where(bw == 0)
                        if len(ys) < 10:
                            continue
                        crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                        temps.append(crop)
            self._templates[digit] = temps

    def _match_digit(self, patch):
        """Template-match a binary digit patch against all templates. Returns (digit, conf)."""
        if patch.size < 20:
            return None, 0.0

        # Binary from preprocess is INV (text=255, bg=0). Flip to standard.
        patch = 255 - patch

        best_digit = None
        best_score = 0.0

        ph, pw = patch.shape
        for digit, templates in self._templates.items():
            for tmpl in templates:
                th, tw = tmpl.shape
                if tw < 5 or th < 5:
                    continue
                # Resize template to match patch dimensions
                scale_w = pw / tw
                scale_h = ph / th
                if scale_w < 0.3 or scale_w > 5.0 or scale_h < 0.3 or scale_h > 5.0:
                    continue
                scaled = cv2.resize(tmpl, (pw, ph))
                result = cv2.matchTemplate(patch, scaled, cv2.TM_CCOEFF_NORMED)
                score = float(np.max(result))
                if score > best_score:
                    best_score = score
                    best_digit = digit

        conf = max(0.0, min(1.0, (best_score - 0.3) / 0.7))
        return best_digit, conf

    # ═══════════════════════════════════════════════════════════
    # Camera
    # ═══════════════════════════════════════════════════════════

    def _open_camera(self):
        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        for _ in range(3):
            cap.read()
        self._cap = cap

    def _grab_frame(self):
        self._open_camera()
        if self._cap is None:
            return None
        ok, frame = cap.read() if (cap := self._cap) else (False, None)
        return frame if ok else None

    # ═══════════════════════════════════════════════════════════
    # Preprocessing
    # ═══════════════════════════════════════════════════════════

    def _preprocess(self, gray):
        """Enhance contrast for thin black lines on white bg."""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                tileGridSize=(self.clahe_grid, self.clahe_grid))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return enhanced, binary

    # ═══════════════════════════════════════════════════════════
    # Dash detection
    # ═══════════════════════════════════════════════════════════

    def _detect_dash(self, gray):
        """Check if a dash is passing through the dash ROI strip."""
        x, y, w, h = self.dash_roi
        x = max(0, min(x, gray.shape[1] - 1))
        y = max(0, min(y, gray.shape[0] - 1))
        w = max(1, min(w, gray.shape[1] - x))
        h = max(1, min(h, gray.shape[0] - y))
        roi = gray[y:y + h, x:x + w]

        _, binary = self._preprocess(roi)
        black_pixels = np.count_nonzero(binary)
        total = roi.size
        black_frac = black_pixels / max(total, 1)
        return black_frac

    def _update_dash_count(self, black_frac):
        """State machine: dash enters ROI → count +1 when it exits."""
        threshold = self.dash_black_ratio
        was_inside = self._dash_inside

        if black_frac > threshold and not was_inside:
            self._dash_debounce += 1
            if self._dash_debounce >= self.dash_debounce_frames:
                self._dash_inside = True
                self._dash_debounce = 0
        elif black_frac < threshold * 0.5 and was_inside:
            self._dash_debounce += 1
            if self._dash_debounce >= self.dash_debounce_frames:
                self._dash_inside = False
                self._dash_count += 1
                self._dash_debounce = 0
        else:
            self._dash_debounce = max(0, self._dash_debounce - 1)

    # ═══════════════════════════════════════════════════════════
    # Digit recognition
    # ═══════════════════════════════════════════════════════════

    def _find_digit_contours(self, binary, frame_w, frame_h):
        """Find digit candidates in binary image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        digits = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > frame_w * frame_h * 0.6:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + w >= frame_w - 1 or y + h >= frame_h - 1:
                continue
            aspect = h / max(w, 1)
            if aspect < 0.3 or aspect > 3.5:
                continue
            digits.append((x, y, w, h))
        digits.sort(key=lambda d: d[0])
        return digits

    def _read_digits(self, binary):
        """Extract and classify digits in frame. Returns (number, conf)."""
        x, y, w, h = self.digit_roi
        roi_bin = binary[y:y + h, x:x + w]

        digit_boxes = self._find_digit_contours(roi_bin, w, h)
        if not digit_boxes:
            return None, 0.0

        digits = []
        confs = []
        for bx, by, bw, bh in digit_boxes:
            mx = max(0, bx - 2)
            my = max(0, by - 2)
            mw = min(w - mx, bw + 4)
            mh = min(h - my, bh + 4)
            patch = roi_bin[my:my + mh, mx:mx + mw]
            d, c = self._match_digit(patch)
            if d is not None:
                digits.append(d)
                confs.append(c)

        if not digits:
            return None, 0.0

        if len(digits) == 1:
            number = digits[0]
        else:
            number = digits[0] * 10 + digits[-1]
        avg_conf = float(np.mean(confs))
        return number, avg_conf

    # ═══════════════════════════════════════════════════════════
    # Main update
    # ═══════════════════════════════════════════════════════════

    def update(self, frame=None):
        """Process one frame. Returns status dict."""
        if frame is None:
            frame = self._grab_frame()
        if frame is None:
            return {"error": "no frame"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced, binary = self._preprocess(gray)

        # Dash detection
        black_frac = self._detect_dash(enhanced)
        self._update_dash_count(black_frac)

        # Digit recognition
        number, conf = self._read_digits(binary)
        if number is not None and conf > 0.5:
            if number != self._last_number:
                if self._last_number is not None and self._dash_count > 0:
                    self.dashes_per_number = self._dash_count
                self._dash_count = 0
                self._last_number = number

        return {
            "dash_count": self._dash_count,
            "dash_inside": self._dash_inside,
            "black_frac": black_frac,
            "number": self._last_number,
            "number_conf": conf if number is not None else 0.0,
            "dashes_per_number": self.dashes_per_number,
        }


# ═══════════════════════════════════════════════════════════
# Standalone demo
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Downward digit reader demo")
    parser.add_argument("--cam", type=int, default=1, help="Camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--image", type=str, default=None,
                        help="Test on a single image instead of camera")
    args = parser.parse_args()

    reader = DownwardReader(cam_idx=args.cam, cam_w=args.width, cam_h=args.height)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Failed to read {args.image}")
            return
        result = reader.update(frame)
        print(f"Result: {result}")
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        return

    print(f"Downward digit reader — camera {args.cam} ({args.width}x{args.height})")
    print("Press ESC to quit")

    fps_t0 = time.time()
    fps_n = 0
    while True:
        frame = reader._grab_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        result = reader.update(frame)
        fps_n += 1
        fps_val = fps_n / max(time.time() - fps_t0, 1e-3) if fps_n % 30 == 0 else 0

        disp = frame.copy()
        rx, ry, rw, rh = reader.dash_roi
        cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)

        status = (f"dash={result['dash_count']} in={result['dash_inside']} "
                  f"bf={result['black_frac']:.2f}")
        if result["number"] is not None:
            status += f" | NUM={result['number']} c={result['number_conf']:.2f}"
        if result.get("dashes_per_number"):
            status += f" | dpn={result['dashes_per_number']}"
        if fps_val:
            status += f" | FPS={fps_val:.0f}"

        cv2.putText(disp, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
        cv2.imshow("Downward Digit Reader", disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
