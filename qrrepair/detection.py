"""Independent QR region proposals. Decoding failure is not detection failure."""
from dataclasses import dataclass, field
import time

import cv2
import numpy as np
import zxingcpp


@dataclass
class Detection:
    corners: np.ndarray
    backend: str
    text: str = ""
    payload: bytes = b""
    decoded: bool = False
    alternatives: list = field(default_factory=list)
    score: float = 0.0


def ordered_quad(points):
    p = np.asarray(points, np.float32).reshape(4, 2)
    if not np.isfinite(p).all():
        return None
    center = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(1))), axis=0)
    if not cv2.isContourConvex(p) or abs(cv2.contourArea(p)) < 64:
        return None
    return p


def detect_regions(image, cancel=lambda: False, seconds=3.0):
    """Bounded cascade, preserving separate symbols and alternate geometries.

    Coordinates are always full resolution. Detector pixel-edge conventions are
    deliberately not guessed here: structural refinement resolves their offsets.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = np.ascontiguousarray(gray)
    deadline = time.monotonic() + seconds
    candidates = []

    def stopped():
        return cancel() or time.monotonic() > deadline

    def add(points, backend, result=None):
        p = ordered_quad(points)
        if p is None:
            return
        d = Detection(p, backend)
        if result is not None and result.valid:
            d.text, d.payload, d.decoded = result.text, bytes(result.bytes), True
        for old in candidates:
            area = min(abs(cv2.contourArea(p)), abs(cv2.contourArea(old.corners)))
            intersection = cv2.intersectConvexConvex(p, old.corners)[0]
            if intersection / max(area, 1) > .85 and np.linalg.norm(p.mean(0)-old.corners.mean(0)) < np.sqrt(area)*.15:
                if np.max(np.linalg.norm(p-old.corners, axis=1)) > .25:
                    old.alternatives.append(d)
                if d.decoded and not old.decoded:
                    old.text, old.payload, old.decoded = d.text, d.payload, True
                return
        candidates.append(d)

    def zxing(src, scale=1.0):
        for result in zxingcpp.read_barcodes(src, formats=zxingcpp.BarcodeFormat.QRCode,
                                            return_errors=True, try_invert=True):
            p = result.position
            add(np.float32([(p.top_left.x, p.top_left.y), (p.top_right.x, p.top_right.y),
                           (p.bottom_right.x, p.bottom_right.y), (p.bottom_left.x, p.bottom_left.y)]) / scale,
                "ZXing" if result.valid else "ZXing 校验错误", result)

    if stopped():
        return []
    zxing(gray)
    ratio = min(1., 1800/max(gray.shape))
    small = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA) if ratio < 1 else gray
    aruco = cv2.QRCodeDetectorAruco()

    def cv_detect(detector, src, scale, name):
        if stopped():
            return
        ok, points = detector.detectMulti(src)
        if ok and points is not None:
            for p in points:
                add(p/scale, name)

    # Always obtain independent geometry, even for decodable symbols.
    cv_detect(aruco, small, ratio, "Aruco")
    if not candidates and not stopped():
        cv_detect(aruco, 255-small, ratio, "Aruco 反色")
    if not candidates and not stopped():
        enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(small)
        cv_detect(aruco, enhanced, ratio, "Aruco 局部对比度")
        if not candidates and not stopped():
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 5)
            cv_detect(aruco, binary, ratio, "Aruco 自适应阈值")
    if not candidates and not stopped():
        cv_detect(cv2.QRCodeDetector(), small, ratio, "OpenCV")
    # Small modules can disappear in the coarse image; inspect original tiles.
    if not candidates and ratio < 1 and not stopped():
        side, overlap = 1500, 250
        for y in range(0, gray.shape[0], side-overlap):
            for x in range(0, gray.shape[1], side-overlap):
                if stopped():
                    break
                tile = gray[y:y+side, x:x+side]
                if min(tile.shape) < 32:
                    continue
                ok, points = aruco.detectMulti(tile)
                if ok:
                    for p in points:
                        add(p + np.float32([x, y]), "Aruco 原图分块")
            if stopped():
                break
    return candidates
