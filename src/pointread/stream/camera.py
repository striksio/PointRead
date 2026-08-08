import time
import threading
import numpy as np
import cv2

import json

from pointread.gesture.pinch import PinchDetector
import os
import datetime

from rapidocr_onnxruntime import RapidOCR

OCR_DIR = "/models/ocr"
ocr_engine = RapidOCR(
    det_model_path=OCR_DIR + "/detection/v5/det.onnx",
    rec_model_path=OCR_DIR + "/languages/english/rec.onnx",
    rec_keys_path=OCR_DIR + "/languages/english/dict.txt",
)
PIPELINE = (
    "nvarguscamerasrc wbmode=1 "
    "! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 "
    "! nvvidconv flip-method=1 ! video/x-raw, format=BGRx "
    "! queue ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
)
SAVE_DIR = "/workspace/assets/captures"

# shared frame buffer, read by the web server
lock = threading.Lock()
latest = {"jpg": None}
# reset flag, set by the web server on page load
state = {"reset": False}
# beep flag, read by the web server and pushed to the browser
signal = {"beep": False}
ocr_out = {"text": None}

WAIT = 2.0          # seconds between activation and the start of drawing
DWELL_R = 20        # pixels, how far the tip may wander and still count as still
DWELL_T = 2.0       # seconds the tip must stay within DWELL_R to auto-stop


def _save_capture(clean, bands):
    if not bands:
        return
    os.makedirs(SAVE_DIR, exist_ok=True)

    xs1 = [b["start"] for b in bands]
    xs2 = [b["fill"] for b in bands]
    ys1 = [b["y"] for b in bands]
    ys2 = [b["y"] + b["h"] for b in bands]

    x1 = max(0, min(xs1))
    x2 = min(clean.shape[1], max(xs2))
    y1 = max(0, min(ys1))
    y2 = min(clean.shape[0], max(ys2))
    if x2 <= x1 or y2 <= y1:
        return

    out = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
    for b in bands:
        bx1 = max(0, b["start"])
        bx2 = min(clean.shape[1], b["fill"])
        by1 = max(0, b["y"])
        by2 = min(clean.shape[0], b["y"] + b["h"])
        if bx2 <= bx1 or by2 <= by1:
            continue
        patch = clean[by1:by2, bx1:bx2]
        out[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = patch

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path = os.path.join(SAVE_DIR, ts + ".png")
    cv2.imwrite(img_path, out)

    # OCR the array directly
    result, _ = ocr_engine(out)
    lines = [r[1] for r in result] if result else []
    text = " ".join(lines)
    print("OCR:", text if text else "(no text)")
    ocr_out["text"] = text

    # save the recognized text next to the image
    meta = {
        "timestamp": ts,
        "image": os.path.basename(img_path),
        "text": text,
        "lines": lines,
    }
    json_path = os.path.join(SAVE_DIR, ts + ".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved", img_path, "and", os.path.basename(json_path))
    
def beep():
    signal["beep"] = True


def _in_bounds(pt, w, h, margin=25):
    x, y = pt
    return margin <= x < w - margin and margin <= y < h - margin


def _crop_square(frame):
    h, w = frame.shape[:2]
    if h > w:
        y0 = (h - w) // 2
        return frame[y0:y0 + w, :]
    x0 = (w - h) // 2
    return frame[:, x0:x0 + h]


def _draw_point(frame, x, y, color, radius):
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), radius, color, -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    cv2.circle(frame, (x, y), 6, color, -1)
    return frame


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _line_binary(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    b = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    inv = cv2.bitwise_not(b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    return cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)


def _line_above(closed, tip, max_gap=120):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(closed, 8)
    tx, ty = tip
    best, best_dy = None, None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        if not (x <= tx <= x + w):
            continue
        bottom = y + h
        dy = ty - bottom
        if 0 <= dy <= max_gap:
            if best_dy is None or dy < best_dy:
                best_dy, best = dy, (x, y, w, h)
    return best


def capture_loop(hand, detector=None):
    if detector is None:
        detector = PinchDetector()

    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    t, n, fps = time.time(), 0, 0.0
    R = detector.r

    capture_at = None
    waiting = False
    recording = False
    anchor = None
    dwell_at = None
    bands = []          # each: {"y", "h", "start", "fill"}

    while True:
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005); continue
            frame = _crop_square(frame)
            clean = frame.copy()

            if state["reset"]:
                capture_at = None
                waiting = False
                recording = False
                anchor = None
                dwell_at = None
                bands = []
                detector.active = False
                detector.armed_at = None
                detector.was_over = False
                state["reset"] = False

            kpts, scores = hand(frame)

            tip = thb = None
            if len(kpts) > 0:
                k, sc = kpts[0], scores[0]
                H, W = frame.shape[:2]
                if sc[8] >= 0.3:
                    p = (int(k[8][0]), int(k[8][1]))
                    if _in_bounds(p, W, H):
                        tip = p
                if not recording and sc[4] >= 0.3:
                    p = (int(k[4][0]), int(k[4][1]))
                    if _in_bounds(p, W, H):
                        thb = p
                        frame = _draw_point(frame, p[0], p[1], (0, 180, 255), R)

            event = detector.update(tip, None if recording else thb)
            if event == "on":
                capture_at = time.time()
                waiting = True
                recording = False
                anchor = None
                dwell_at = None
                bands = []
                print("capture start")

            # after the wait, beep once and begin recording
            if waiting and time.time() - capture_at >= WAIT:
                waiting = False
                recording = True
                anchor = None
                dwell_at = None
                bands = []
                beep()

            # record the fingertip, accumulate line bands, run the dwell check
            if recording and tip is not None:
                tx, ty = tip

                # is the tip within the current active band's vertical range
                cur = bands[-1] if bands else None
                on_current = (cur is not None and
                              cur["y"] - 10 <= ty <= cur["y"] + cur["h"] + 30)

                if not on_current:
                    # detect the line above the tip and lock it as a new band
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    closed = _line_binary(gray)
                    box = _line_above(closed, tip)
                    if box is not None:
                        x, y, w, h = box
                        pad = int(h * 0.15)
                        y = y - pad
                        h = h + 2 * pad
                        start = tx if not bands else x  # first line at finger, later at left edge
                        bands.append({"x": x, "w": w, "y": y, "h": h,
                                      "start": start, "fill": start})
                        cur = bands[-1]

                # extend the current band fill to the fingertip x
                if cur is not None:
                    cur["fill"] = max(cur["fill"], min(tx, cur["x"] + cur["w"]))

                # dwell-to-stop
                if anchor is None or _dist(tip, anchor) > DWELL_R:
                    anchor = tip
                    dwell_at = time.time()
                elif time.time() - dwell_at >= DWELL_T:
                    recording = False
                    detector.active = False
                    detector.armed_at = None
                    detector.was_over = False
                    _save_capture(clean, bands)
                    beep()
                    print("capture stop (dwell)")

            # draw all locked bands as translucent red fill
            if bands:
                overlay = frame.copy()
                for b in bands:
                    cv2.rectangle(overlay, (b["start"], b["y"]),
                                  (b["fill"], b["y"] + b["h"]), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # countdown during the wait
            if waiting:
                left = WAIT - (time.time() - capture_at)
                if left > 0:
                    cv2.putText(frame, f"{left:.1f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            label = "ON" if detector.active else "OFF"
            col = (0, 255, 0) if detector.active else (0, 0, 255)
            cv2.putText(frame, label, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

            n += 1
            if n % 15 == 0:
                fps = 15 / (time.time() - t); t = time.time()
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with lock:
                    latest["jpg"] = jpg.tobytes()
        except Exception as e:
            print("capture err:", e); time.sleep(0.01)
