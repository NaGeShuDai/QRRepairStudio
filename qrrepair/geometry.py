"""Module-coordinate geometry, masked structural registration and sampling."""
from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np


@dataclass
class GeometryQuality:
    residual: float = 1.0
    coverage: float = 0.0
    movement: float = 0.0
    refined: bool = False
    ambiguous: bool = False
    message: str = ""


def homography(corners, n):
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]),
                                       np.asarray(corners, np.float32))


@lru_cache(maxsize=80)
def module_points(n, dense=False):
    # Match image pixel centers: the module interval [0,1) has center 0.5.
    offsets = np.array([.25, .5, .75] if dense else [.5], np.float32)
    axis = (np.arange(n, dtype=np.float32)[:, None] + offsets).ravel()
    x, y = np.meshgrid(axis, axis)
    return np.stack([x, y], -1).reshape(-1, 1, 2)


def sample_evidence(gray, corners, version, bias=0, dense=True):
    n = 17+4*version
    k = 3 if dense else 1
    pts = cv2.perspectiveTransform(module_points(n, dense), homography(corners, n)).reshape(n*k, n*k, 2)
    x, y = pts[..., 0], pts[..., 1]
    values = cv2.remap(gray, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255).astype(np.float32)
    valid = (x >= 0) & (y >= 0) & (x <= gray.shape[1]-1) & (y <= gray.shape[0]-1)
    cells = values.reshape(n, k, n, k).transpose(0, 2, 1, 3)
    covered = valid.reshape(n, k, n, k).transpose(0, 2, 1, 3)
    # The center carries most information under severe blur; off-center samples
    # estimate stability, not a vote that lets neighboring modules dominate.
    intensity = cells[:, :, k//2, k//2] if dense else np.median(cells, axis=(2, 3))
    observed = values[valid]
    low, high = np.percentile(observed, [12, 88]) if observed.size else (0, 255)
    contrast = max(float(high-low), 24.)
    threshold = .75*(low+high)/2 + .25*cv2.GaussianBlur(intensity, (0, 0), 3) + bias
    p = np.clip(.5 + (threshold-intensity)/contrast, .015, .985)
    coverage = covered.mean(axis=(2, 3))
    if dense:
        # Spatial disagreement under +/- quarter-module sampling reduces certainty.
        local_p = np.clip(.5+(threshold[..., None, None]-cells)/contrast, .015, .985)
        instability = local_p.max(axis=(2, 3))-local_p.min(axis=(2, 3))
        reliability = coverage * (1-.8*instability)
    else:
        reliability = coverage
    p = .5+(p-.5)*reliability
    return p.astype(np.float32), coverage.astype(np.float32)


def _structural_image(gray, corners, version, inverted, scale=4):
    from .engine import template, fixed_mask
    n = 17+4*version
    size = n*scale
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    transform = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    aligned = cv2.warpPerspective(gray, transform, (size, size), flags=cv2.INTER_LINEAR,
                                  borderValue=255)
    if inverted:
        aligned = 255-aligned
    fixed = fixed_mask(version)
    expected = np.where(template(version) == 1, 0, 255).astype(np.uint8)
    target = np.repeat(np.repeat(expected, scale, 0), scale, 1)
    # Quarter-pixel antialias phase avoids registering to the last black pixel
    # rather than the continuous boundary at the structural transition.
    target = cv2.warpAffine(target, np.float32([[1,0,.25],[0,1,.25]]),
                            (size,size), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = np.repeat(np.repeat(fixed.astype(np.uint8)*255, scale, 0), scale, 1)
    support = cv2.warpPerspective(np.ones(gray.shape, np.uint8)*255, transform, (size, size))
    # Exclude unknown data cells on both sides of the registration, never fill
    # those cells with guesses that could pull the geometry toward a wrong code.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return aligned, target, mask, support, transform


def structural_error(gray, corners, version, inverted=False):
    from .engine import template, fixed_mask
    p, coverage = sample_evidence(gray, corners, version, dense=True)
    if inverted:
        p = 1-p
    fixed = fixed_mask(version)
    loss = np.where(template(version) == 1, 1-p, p)
    return float(loss[fixed].mean()), float(coverage[fixed].mean())


def refine_geometry(gray, corners, version, inverted=False, cancel=lambda: False, locked=None):
    """Two bounded, masked homography fits; original-image evidence gates updates.

    corners are in QR orientation (may have mirrored winding). Unobserved pixels
    never become white evidence. Locked corners are projected back exactly.
    """
    from .engine import validate_quad
    initial = np.asarray(corners, np.float32).copy()
    n = 17+4*version
    before, coverage = structural_error(gray, initial, version, inverted)
    quality = GeometryQuality(before, coverage)
    if cancel() or before > .34 or coverage < .65:
        quality.message = "结构证据不足，请调整选区"
        return initial, quality
    current, best = initial.copy(), before
    pitch = max(np.mean(np.linalg.norm(np.roll(initial, -1, axis=0)-initial, axis=1))/n, 1.)
    locks = np.zeros(4, bool) if locked is None else np.asarray(locked, bool)
    for _ in range(2):
        if cancel():
            break
        aligned, target, mask, support, transform = _structural_image(gray, current, version, inverted)
        # Normalize slow illumination variation; preserve module transitions.
        def normalize(a):
            f = a.astype(np.float32)/255
            return np.ascontiguousarray(f-cv2.GaussianBlur(f, (0, 0), 12)+.5)
        motion = np.eye(3, dtype=np.float32)
        try:
            if hasattr(cv2, "findTransformECCWithMask"):
                _, motion = cv2.findTransformECCWithMask(normalize(target), normalize(aligned),
                    mask, support, motion, cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 1e-4), 5)
            else:
                _, motion = cv2.findTransformECC(normalize(target), normalize(aligned), motion,
                    cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 1e-4),
                    cv2.bitwise_and(mask, support), 5)
            size = len(target)
            square = np.float32([[[0, 0], [size, 0], [size, size], [0, size]]])
            proposed = cv2.perspectiveTransform(square, np.linalg.inv(transform) @ motion)[0]
            proposed[locks] = initial[locks]
            validate_quad(proposed)
            movement = np.max(np.linalg.norm(proposed-initial, axis=1))/pitch
            if movement > 1.75:
                break
            loss, observed = structural_error(gray, proposed, version, inverted)
            if loss >= best-.0005 or observed < coverage-.01:
                break
            current, best = proposed, loss
            quality = GeometryQuality(loss, observed, float(movement), True)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            break
    quality.message = "边界已精修" if quality.refined else "边界已核验"
    if quality.residual > .23:
        quality.message = "局部结构偏差较大，可固定角点后重试"
    return current, quality
