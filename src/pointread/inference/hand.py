import numpy as np
import cv2

from pointread.inference.engine import Engine

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class TRTHand:
    """RTMDet hand detector followed by RTMPose landmark model.

    Returns keypoints for the single highest-confidence hand.
    Landmark 8 is the index fingertip, landmark 4 is the thumb tip.
    """

    def __init__(self, det_path, pose_path, det_thr=0.35, max_det=100):
        self.det = Engine(det_path, (1, 3, 320, 320), max_out=max_det)
        self.pose = Engine(pose_path, (1, 3, 256, 256))
        self.det_thr = det_thr

    def _prep(self, img, size):
        h, w = img.shape[:2]
        s = min(size / w, size / h)
        nw, nh = int(w * s), int(h * s)
        r = cv2.resize(img, (nw, nh))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = r
        x = ((canvas.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), s

    def detect(self, img):
        x, s = self._prep(img, 320)
        o = self.det.infer(x)
        dets = o["dets"]
        if dets.ndim == 3:
            dets = dets[0]
        H, W = img.shape[:2]
        out = []
        for d in dets:
            if d[4] < self.det_thr:
                continue
            b = d[:4] / s
            b[0] = max(0, min(W - 1, b[0])); b[2] = max(0, min(W, b[2]))
            b[1] = max(0, min(H - 1, b[1])); b[3] = max(0, min(H, b[3]))
            out.append((b, float(d[4])))
        return out

    def landmarks(self, img, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max((x2 - x1), (y2 - y1)) * 1.25
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = int(cx + side / 2), int(cy + side / 2)
        H, W = img.shape[:2]
        px1, py1, px2, py2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            return np.zeros((21, 2)), np.zeros(21)
        ch, cw = crop.shape[:2]
        r = cv2.resize(crop, (256, 256))
        xin = ((r.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        o = self.pose.infer(np.ascontiguousarray(xin))
        sx, sy = o["simcc_x"][0], o["simcc_y"][0]
        kx = sx.argmax(1) / 2.0
        ky = sy.argmax(1) / 2.0
        scores = (sx.max(1) + sy.max(1)) / 2
        kpts = np.stack([kx / 256 * cw + px1, ky / 256 * ch + py1], 1)
        return kpts, scores

    def __call__(self, img):
        dets = self.detect(img)
        if not dets:
            return np.zeros((0, 21, 2)), np.zeros((0, 21))
        box, score = max(dets, key=lambda bs: bs[1])
        k, s = self.landmarks(img, box)
        return k[None], s[None]
