import time
import threading
import subprocess

import cv2

from pointread.gesture.pinch import PinchDetector

PIPELINE = (
    "nvarguscamerasrc wbmode=1 "
    "! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 "
    "! nvvidconv flip-method=1 ! video/x-raw, format=BGRx "
    "! queue ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
)

# shared frame buffer, read by the web server
lock = threading.Lock()
latest = {"jpg": None}
# reset flag, set by the web server on page load
state = {"reset": False}

WAIT = 2.0  # seconds between activation and the start of drawing


# near the shared state
signal = {"beep": False}

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


def capture_loop(hand, detector=None):
    if detector is None:
        detector = PinchDetector()

    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    t, n, fps = time.time(), 0, 0.0
    R = detector.r

    trail = []
    capture_at = None
    waiting = False
    recording = False

    while True:
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005); continue
            frame = _crop_square(frame)

            if state["reset"]:
                trail = []
                capture_at = None
                waiting = False
                recording = False
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
                        frame = _draw_point(frame, p[0], p[1], (0, 255, 0), R)
                if sc[4] >= 0.3:
                    p = (int(k[4][0]), int(k[4][1]))
                    if _in_bounds(p, W, H):
                        thb = p
                        frame = _draw_point(frame, p[0], p[1], (0, 180, 255), R)

            event = detector.update(tip, thb)
            if event == "on":
                trail = []
                capture_at = time.time()
                waiting = True
                recording = False
                print("capture start")
            elif event == "off":
                waiting = False
                recording = False
                print("capture stop")

            # after the wait, beep once and begin recording
            if waiting and time.time() - capture_at >= WAIT:
                waiting = False
                recording = True
                beep()

            # record the fingertip only while recording
            if recording and tip is not None:
                trail.append(tip)

            # countdown during the wait
            if waiting:
                left = WAIT - (time.time() - capture_at)
                if left > 0:
                    cv2.putText(frame, f"{left:.1f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # draw the trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 4)
            for p in trail:
                cv2.circle(frame, p, 8, (0, 255, 0), -1)

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
