from typing import List, Optional

import numpy as np

from ..core.types import Detection
from .runtime import make_runner


def letterbox(img: np.ndarray, size: int, pad_value=114):
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


class YoloxPersonDetector:
    """YOLOX (Apache-2.0) exported with tools/export_onnx.py, i.e. raw head output
    (1, N, 5 + num_classes) without decode. Person = COCO class 0. Input is BGR uint8."""

    def __init__(self, model_path: str, input_size: int = 640, score_thresh: float = 0.1,
                 nms_thresh: float = 0.45, person_class: int = 0):
        self.runner = make_runner(model_path)
        self.size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.person_class = person_class
        self._grids = self._make_grids(input_size)

    @staticmethod
    def _make_grids(size: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> List[Detection]:
        img, r = letterbox(bgr, self.size)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32))
        pred = self.runner(x)[0][0]
        grids, strides = self._grids
        xy = (pred[:, :2] + grids) * strides
        wh = np.exp(pred[:, 2:4]) * strides
        scores = pred[:, 4] * pred[:, 5 + self.person_class]
        m = scores >= self.score_thresh
        if not m.any():
            return []
        xy, wh, scores = xy[m], wh[m], scores[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / r
        h, w = bgr.shape[:2]
        boxes = np.clip(boxes, 0, [w, h, w, h])
        keep = nms(boxes, scores, self.nms_thresh)
        # Low-score boxes are kept on purpose: ByteTrack uses them to bridge occlusion.
        return [Detection(box=boxes[i], score=float(scores[i])) for i in keep]


class OsnetReid:
    """OSNet / fast-reid ONNX or TRT model, input (1, 3, 256, 128) RGB normalised, output (1, D)."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str, height: int = 256, width: int = 128):
        self.runner = make_runner(model_path)
        self.h, self.w = height, width

    def __call__(self, bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        import cv2

        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = np.clip(box, 0, [W, H, W, H]).astype(int)
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        crop = cv2.resize(bgr[y1:y2, x1:x2], (self.w, self.h))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((crop - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        return self.runner(np.ascontiguousarray(x, dtype=np.float32))[0].ravel()


class RtmPose:
    """RTMPose (Apache-2.0) SimCC ONNX via rtmlib, top-down on given boxes. Optional."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from rtmlib import RTMPose

        self.model = RTMPose(model_path, model_input_size=(192, 256), backend="onnxruntime", device=device)

    def __call__(self, bgr: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
        if not boxes:
            return []
        kps, scores = self.model(bgr, bboxes=[b.tolist() for b in boxes])
        return [np.concatenate([k, s[:, None]], axis=1) for k, s in zip(kps, scores)]
