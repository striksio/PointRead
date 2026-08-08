import threading

from aiohttp import web

from pointread.inference.hand import TRTHand
from pointread.gesture.pinch import PinchDetector
from pointread.stream.camera import capture_loop
from pointread.stream.server import build_app

CKPT = "/models/engines/"
DET_ENGINE = CKPT + "rtmdet_hand_fp16.engine"
POSE_ENGINE = CKPT + "rtmpose_hand_fp16.engine"


def main():
    hand = TRTHand(DET_ENGINE, POSE_ENGINE)
    detector = PinchDetector()

    threading.Thread(
        target=capture_loop, args=(hand, detector), daemon=True
    ).start()

    app = build_app()
    web.run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
