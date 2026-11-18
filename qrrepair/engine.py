from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps
import qrcode
from qrcode import base, util
from reedsolo import RSCodec, ReedSolomonError
import zxingcpp

EC_NAMES = {1: "L", 0: "M", 3: "Q", 2: "H"}


@dataclass
class Grid:
    values: np.ndarray  # -1 unknown, 0 white, 1 black
    probability: np.ndarray  # relative black evidence, not calibrated probability
    locked: np.ndarray
    source: np.ndarray  # 0 sampled, 1 manual, 2 structural, 3 RS

    @property
    def version(self):
        return (self.values.shape[0] - 17) // 4

    def copy(self):
        return Grid(*(x.copy() for x in (self.values, self.probability, self.locked, self.source)))


@dataclass
class Sample:
    grid: Grid
    score: float
    rotation: int = 0
    mirrored: bool = False
    inverted: bool = False
    corners: np.ndarray | None = None  # refined geometry in original input order
    geometry: object = None


@dataclass
class Repair:
    matrix: np.ndarray
    text: str
    payload: bytes
    ec: str
    mask: int
    score: float
    changed: int
    strong_changed: int
    blocks: list[str]
    evidence: Grid | None = None
    corners: np.ndarray | None = None


def read_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.width * img.height > 60_000_000:
            raise ValueError("图片超过 6000 万像素，请先缩小。")
        # Transparent backgrounds must be composited over white, not converted to black.
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        return np.array(bg.convert("RGB"))


def render_matrix(matrix, scale=10, border=4):
    gray = np.where(matrix < 0, 160, np.where(matrix == 1, 0, 255)).astype(np.uint8)
    return np.repeat(np.repeat(np.pad(gray, border, constant_values=255), scale, 0), scale, 1)


def decode_image(image, pure=False):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return zxingcpp.read_barcodes(np.ascontiguousarray(image), formats=zxingcpp.BarcodeFormat.QRCode,
                                 try_rotate=True, try_invert=True, is_pure=pure)


def validate_quad(points):
    p = np.asarray(points, np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("需要四个有效角点。")
    if not cv2.isContourConvex(p.reshape(-1, 1, 2)) or abs(cv2.contourArea(p)) < 64:
        raise ValueError("四角必须按顺序围成凸四边形，且面积足够大；请避免交叉。")
    return p


def background_mask(image, color=None, colorful=True, tolerance=38):
    rgb = image.astype(np.int16)
    mask = (rgb.max(2)-rgb.min(2) > 32) if colorful else np.zeros(image.shape[:2],bool)
    if color is not None:
        mask |= np.max(np.abs(rgb-np.asarray(color,dtype=np.int16)),axis=2) <= tolerance
    return mask


def detection_image(image, color=None, colorful=False, inverted=False):
    mask = background_mask(image,color,colorful)
    out = (255-image) if inverted else image.copy()
    out[mask] = 255
    return out


def locate(image, background=None, colorful=False, inverted=False):
    """Compatibility entry point; UI uses locate_candidates to retain symbols."""
    candidates = locate_candidates(image, background, colorful, inverted)
    if not candidates:
        return None, []
    return candidates[0].corners.copy(), [(d.text, d.payload) for d in candidates if d.decoded]


def locate_candidates(image, background=None, colorful=False, inverted=False, cancel=lambda: False):
    from .detection import detect_regions
    candidates = detect_regions(detection_image(image, background, colorful, inverted), cancel)
    # Select detector geometry by original structural evidence; never by backend
    # confidence alone. Keep all distinct symbols for explicit UI selection.
    for candidate in candidates:
        if cancel():
            break
        proposals = [candidate] + candidate.alternatives
        best, best_score = candidate.corners, -1.
        for proposal in proposals[:4]:
            samples = scan(image, proposal.corners, invert=inverted, cancel=cancel, refine=False)
            if samples and samples[0].score > best_score:
                best, best_score = proposal.corners, samples[0].score
        candidate.corners = best.copy()
        candidate.score = best_score
    return sorted([d for d in candidates if d.score >= .68], key=lambda d: (d.decoded, d.score), reverse=True)


def optimize_quad(image, corners, background=None, colorful=False, inverted=False, cancel=lambda: False):
    """Detect only inside the rectified user ROI; never fall back to a global scan."""
    p = validate_quad(corners)
    size = int(np.clip(np.max(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)), 256, 1600))
    crop = warp(detection_image(image,background,colorful,inverted), p, size)
    pad = max(24, size // 8)
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    proposals = locate_candidates(padded, cancel=cancel)
    if not proposals:
        return None, []
    # Prefer the symbol occupying the user's ROI, not decoder result order.
    proposals.sort(key=lambda d: abs(cv2.contourArea(d.corners)), reverse=True)
    valid = [d for d in proposals if np.all(d.corners-pad >= -2) and np.all(d.corners-pad <= size+2)]
    if not valid:
        return None, []
    chosen = valid[0]
    contents = [(chosen.text, chosen.payload)] if chosen.decoded else []
    local = chosen.corners - pad
    # Permit only detector rounding at the padded crop boundary.
    if np.any(local < -2) or np.any(local > size+2):
        return None, []
    local = np.clip(local, 0, size).astype(np.float32)
    square = np.float32([[0,0],[size,0],[size,size],[0,size]])
    inverse = cv2.getPerspectiveTransform(square, p)
    mapped = cv2.perspectiveTransform(local[None], inverse)[0]
    h, w = image.shape[:2]
    mapped[:,0] = np.clip(mapped[:,0], 0, w)
    mapped[:,1] = np.clip(mapped[:,1], 0, h)
    try:
        return validate_quad(mapped), contents
    except ValueError:
        return None, []


@lru_cache(maxsize=2048)
def template(version, ec=0, mask=0):
    """Use python-qrcode's tested function patterns and version tables."""
    q = qrcode.QRCode(version=version, error_correction=ec, border=0, mask_pattern=mask)
    n = 17 + version * 4
    q.modules_count = n
    q.modules = [[None] * n for _ in range(n)]
    q.setup_position_probe_pattern(0, 0)
    q.setup_position_probe_pattern(n - 7, 0)
    q.setup_position_probe_pattern(0, n - 7)
    q.setup_position_adjust_pattern()
    q.setup_timing_pattern()
    q.setup_type_info(False, mask)
    if version >= 7:
        q.setup_type_number(False)
    arr = np.array([[(-1 if v is None else int(v)) for v in row] for row in q.modules], np.int8)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=40)
def fixed_mask(version):
    # Exclude BOTH format copies, whose values are hypotheses, from geometry scoring.
    t = template(version)
    result = t >= 0
    for ec in EC_NAMES:
        for m in range(8):
            result &= template(version, ec, m) == t
    result.flags.writeable = False
    return result


@lru_cache(maxsize=40)
def finder_mask(version):
    n = 17 + 4*version
    mask = np.zeros((n,n), bool)
    mask[:8,:8] = mask[:8,-8:] = mask[-8:,:8] = True
    mask.flags.writeable = False
    return mask


@lru_cache(maxsize=40)
def region_masks(version):
    """Return fixed QR function-module regions for the editor overlay."""
    n = 17 + 4 * version
    t = template(version)
    finder = np.zeros((n, n), bool)
    finder[:8, :8] = finder[:8, -8:] = finder[-8:, :8] = True
    separator = finder.copy()
    finder[:] = False
    finder[:7,:7] = finder[:7,-7:] = finder[-7:,:7] = True
    separator &= ~finder
    timing = np.zeros((n, n), bool)
    timing[6, 8:n-8] = True
    timing[8:n-8, 6] = True
    format_mask = np.zeros((n, n), bool)
    format_mask[8, :9] = True
    format_mask[:9, 8] = True
    format_mask[8, n-8:n] = True
    format_mask[n-7:n, 8] = True
    alignment = (t >= 0) & ~finder & ~separator & ~timing & ~format_mask
    # Version information is present only for versions 7–40 and is identifiable
    # as the two 3×6 fixed blocks near the upper-right/lower-left finders.
    version_info = np.zeros((n, n), bool)
    if version >= 7:
        version_info[:6, n-11:n-8] = True
        version_info[n-11:n-8, :6] = True
    alignment &= ~version_info
    data = ~(finder | separator | timing | format_mask | version_info | alignment)
    result = {"分隔": separator, "定位": finder, "时序": timing, "格式": format_mask,
              "版本": version_info, "校正": alignment, "数据": data}
    for value in result.values():
        value.flags.writeable = False
    return result


@lru_cache(maxsize=40)
def data_positions(version):
    t = template(version)
    n = len(t)
    row, inc = n - 1, -1
    positions = []
    for col in range(n - 1, 0, -2):
        if col <= 6:
            col -= 1
        while True:
            for c in (col, col - 1):
                if t[row, c] < 0:
                    positions.append((row, c))
            row += inc
            if row < 0 or row >= n:
                row -= inc
                inc *= -1
                break
    return tuple(positions)


def warp(image, corners, size=800):
    p = validate_quad(corners)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    h = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(image, h, (size, size), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def sample_grid(image, corners, version, bias=0, uncertain=0.18):
    from .geometry import sample_evidence
    validate_quad(corners)
    n = version * 4 + 17
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    p, _ = sample_evidence(gray, corners, version, bias)
    matrix = (p >= .5).astype(np.int8)
    return Grid(matrix, p, np.zeros((n, n), bool), np.zeros((n, n), np.uint8))


def scan(image, corners, versions=None, bias=0, uncertain=.18, cancel=lambda: False, progress=lambda s: None,
         mirror=False, invert=False, refine=True, locked_corners=None):
    from .geometry import sample_evidence, refine_geometry
    validate_quad(corners)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    results = []
    sampled = {}
    choices = list(versions) if versions is not None else list(range(1, 41))
    if versions is None:
        # Cheap structural screening avoids constructing 32 format hypotheses for
        # forty implausible dimensions. All dimensions still receive image tests.
        ranks = []
        for v in choices:
            if cancel():
                return []
            p, _ = sample_evidence(gray, corners, v, bias, dense=False)
            sampled[v] = p
            t, f = template(v), finder_mask(v)
            direction_scores = []
            for rotation in range(4):
                pp = np.rot90(np.fliplr(p) if mirror else p, rotation)
                if invert:
                    pp = 1-pp
                loss = np.where(t == 1, 1-pp, pp)
                # Timing is independent of format hypotheses and discriminates pitch.
                timing = np.concatenate([loss[6, 8:-8], loss[8:-8, 6]])
                direction_scores.append(.7*loss[f].mean()+.3*timing.mean())
            ranks.append((min(direction_scores), v))
        choices = [v for _, v in sorted(ranks)[:10]]
    for v in choices:
        if cancel():
            break
        progress(f"采样版本 {v} / 40")
        if v in sampled:
            p = sampled[v]
        else:
            p, _ = sample_evidence(gray, corners, v, bias, dense=False)
        val = (p >= .5).astype(np.int8)
        g = Grid(val, p, np.zeros_like(val, bool), np.zeros_like(val, np.uint8))
        fixed = fixed_mask(v)
        finders = finder_mask(v)
        t = template(v)
        format_cells = (t >= 0) & ~fixed
        formats = np.stack([template(v, ec, mask)[format_cells] for ec in EC_NAMES for mask in range(8)])
        best = None
        # Orientation and negative/mirrored photos are explicit candidate properties.
        for mirrored in (bool(mirror),):
            for rotation in range(4):
                p = np.rot90(np.fliplr(g.probability) if mirrored else g.probability, rotation).copy()
                for inverted in (bool(invert),):
                    prob = 1 - p if inverted else p
                    losses = np.where(t == 1, 1 - prob, prob)
                    finder_loss = losses[finders].mean()
                    structure_loss = losses[fixed & ~finders].mean()
                    # Finder/timing geometry is symmetric under transpose. Both format
                    # copies provide the missing direction evidence before RS decoding.
                    fp = prob[format_cells]
                    format_loss = np.where(formats == 1, 1-fp, fp).mean(axis=1).min()
                    score = float(1 - .55*finder_loss - .30*structure_loss - .15*format_loss)
                    if best is None or score > best.score:
                        val = (prob >= .5).astype(np.int8)
                        gg = Grid(val, prob.copy(), np.zeros_like(val, bool), np.zeros_like(val, np.uint8))
                        best = Sample(gg, score, rotation, mirrored, inverted)
        # Score the original evidence first; then make invariant function modules
        # explicit in the very first grid, never part of bit-flip search.
        results.append(best)
    results.sort(key=lambda x: x.score, reverse=True)
    if refine:
        # Compare candidates using the same dense reliability model. A coarse
        # unrefined runner-up must not outrank a good fit just because its
        # confidence has not yet been discounted for spatial instability.
        for sample in results:
            if cancel():
                break
            oriented = oriented_corners(corners, sample)
            prob, _ = sample_evidence(gray, oriented, sample.grid.version, bias, dense=True)
            if invert:
                prob = 1-prob
            sample.grid = Grid((prob >= .5).astype(np.int8), prob, np.zeros_like(prob, bool), np.zeros_like(prob, np.uint8))
            sample.score = structure_score(prob, sample.grid.version)
        results.sort(key=lambda x: x.score, reverse=True)
        for sample in results[:min(3, len(results))]:
            if cancel():
                break
            progress(f"精修 V{sample.grid.version} 网格边界")
            oriented = oriented_corners(corners, sample)
            locks = None
            if locked_corners is not None:
                locks = np.asarray(locked_corners, bool)
                if sample.mirrored:
                    locks = locks[[1, 0, 3, 2]]
                locks = np.roll(locks, -sample.rotation)
            refined, quality = refine_geometry(gray, oriented, sample.grid.version, invert, cancel, locks)
            original_order = np.roll(refined, sample.rotation, axis=0)
            if sample.mirrored:
                original_order = original_order[[1, 0, 3, 2]]
            sample.corners, sample.geometry = original_order, quality
            prob, _ = sample_evidence(gray, refined, sample.grid.version, bias, dense=True)
            if invert:
                prob = 1-prob
            sample.grid = Grid((prob >= .5).astype(np.int8), prob, np.zeros_like(prob, bool), np.zeros_like(prob, np.uint8))
            sample.score = structure_score(prob, sample.grid.version)
    for sample in results:
        sample.grid = restore_structure(sample.grid)
    results.sort(key=lambda x: x.score, reverse=True)
    if len(results) > 1 and results[0].geometry is not None:
        results[0].geometry.ambiguous = results[0].score-results[1].score < .025
    return results


def structure_score(prob, version):
    t, fixed, finders = template(version), fixed_mask(version), finder_mask(version)
    losses = np.where(t == 1, 1-prob, prob)
    fmt = (t >= 0) & ~fixed
    formats = np.stack([template(version, ec, m)[fmt] for ec in EC_NAMES for m in range(8)])
    format_loss = np.where(formats == 1, 1-prob[fmt], prob[fmt]).mean(1).min()
    return float(1-.55*losses[finders].mean()-.30*losses[fixed & ~finders].mean()-.15*format_loss)


def oriented_corners(corners, sample):
    p = np.array(sample.corners if sample.corners is not None else corners, np.float32)
    if sample.mirrored:
        p = p[[1, 0, 3, 2]]
    return np.roll(p, -sample.rotation, axis=0)


def restore_structure(grid):
    out = grid.copy()
    m = fixed_mask(grid.version) & ~grid.locked
    out.values[m] = template(grid.version)[m]
    out.source[m] = 2
    return out


def resolve_unknowns(grid):
    """Migrate legacy unknown cells to their stronger evidence; never invent certainty."""
    out = grid.copy()
    unknown = out.values < 0
    out.values[unknown] = (out.probability[unknown] >= .5).astype(np.int8)
    out.source[unknown] = 0
    out.locked[unknown] = False
    return out


@lru_cache(maxsize=160)
def block_layout(version, ec):
    blocks = base.rs_blocks(version, ec)
    mapping = [[] for _ in blocks]
    k = 0
    for i in range(max(b.data_count for b in blocks)):
        for b, block in enumerate(blocks):
            if i < block.data_count:
                mapping[b].append(k)
                k += 1
    for i in range(max(b.total_count - b.data_count for b in blocks)):
        for b, block in enumerate(blocks):
            if i < block.total_count - block.data_count:
                mapping[b].append(k)
                k += 1
    return blocks, mapping


def repair(grid: Grid, seconds=12, deep=False, cancel=lambda: False, progress=lambda s: None):
    """Soft evidence guides erasure schedules and a bounded Chase search.

    RS arithmetic and syndrome validation are provided by reedsolo; QR placement,
    masks, block tables and matrix reconstruction are provided by python-qrcode.
    """
    start = time.monotonic()
    deadline = start + seconds
    v = grid.version
    coords = np.array(data_positions(v))
    rows, cols = coords[:, 0], coords[:, 1]
    hard = np.where(grid.values < 0, grid.probability >= .5, grid.values).astype(np.uint8)
    confidence = np.abs(grid.probability - .5) * 2
    confidence = np.where((grid.source == 1) & (grid.values >= 0), .999, confidence)
    fixed = template(v) >= 0
    hypotheses = []
    for ec in EC_NAMES:
        for mask in range(8):
            t = template(v, ec, mask)
            # User-locked function cells can reject a hypothesis immediately.
            if np.any(grid.locked & fixed & (grid.values >= 0) & (grid.values != t)):
                continue
            loss = np.where(t == 1, 1-grid.probability, grid.probability)[fixed].mean()
            hypotheses.append((float(loss), ec, mask))
    hypotheses.sort()
    results = []
    seen = set()
    best_diagnostic = []
    best_passed = -1

    def stopped():
        return cancel() or time.monotonic() > deadline

    for _, ec, mask in hypotheses:
        if stopped():
            break
        progress(f"校验 {EC_NAMES[ec]} / 掩码 {mask}，已有 {len(results)} 个候选")
        blocks, mapping = block_layout(v, ec)
        count = sum(b.total_count for b in blocks)
        mask_bits = np.array([util.mask_func(mask)(int(r), int(c)) for r, c in coords], np.uint8)
        raw_bits = hard[rows, cols] ^ mask_bits
        raw = np.packbits(raw_bits[:count*8]).astype(np.uint8)
        conf = confidence[rows, cols][:count*8].reshape(-1, 8)
        unknown = (grid.values[rows, cols][:count*8].reshape(-1, 8) < 0).any(1)
        corrected = raw.copy()
        diagnostics = []
        passed = 0
        for bi, (block, indices) in enumerate(zip(blocks, mapping)):
            if stopped():
                diagnostics.extend(f"块 {j+1}: 未检查 · 搜索提前结束" for j in range(bi, len(blocks)))
                break
            nsym = block.total_count - block.data_count
            codec = RSCodec(nsym)
            data = raw[indices].copy()
            reliab = conf[indices].min(1)
            uncertain_indices = np.flatnonzero(unknown[indices]).tolist()
            ranked = [int(i) for i in np.argsort(reliab) if reliab[i] < .65 or unknown[indices[i]]]
            schedules = [[], uncertain_indices[:nsym]]
            for amount in (2, 4, 6, nsym//2, nsym):
                if 0 < amount <= nsym and amount <= len(ranked):
                    schedules.append(ranked[:amount])
            answer = None
            changes = 0

            def attempt(trial, erasures):
                try:
                    _, full, _ = codec.decode(bytearray(trial), erase_pos=erasures)
                    if not all(codec.check(full)):
                        return None
                    candidate = np.frombuffer(bytes(full), np.uint8).copy()
                    # Locked data bits are checked on every block, allowing alternative trials.
                    for local, inter in enumerate(indices):
                        if candidate[local] == raw[inter]:
                            continue
                        for bit in range(8):
                            r, c = coords[inter*8+bit]
                            expected = ((int(candidate[local]) >> (7-bit)) & 1) ^ mask_bits[inter*8+bit]
                            if grid.locked[r, c] and grid.values[r, c] >= 0 and expected != grid.values[r, c]:
                                return None
                    return candidate
                except (ReedSolomonError, ValueError, IndexError):
                    return None

            for erasures in schedules:
                if stopped():
                    break
                answer = attempt(data, erasures)
                if answer is not None:
                    break
            if answer is None and deep:
                # Search the least reliable unlocked bits of this failing block.
                flat_conf = conf[indices].reshape(-1)
                bit_order = [int(i) for i in np.argsort(flat_conf) if flat_conf[i] < .65][:10]
                trials = [(int(k),) for k in bit_order]
                trials += list(combinations([int(k) for k in bit_order[:8]], 2))
                for bits in trials:
                    if stopped():
                        break
                    trial = data.copy()
                    for bit in bits:
                        trial[bit//8] ^= np.uint8(1 << (7-bit % 8))
                    answer = attempt(trial, [])
                    if answer is not None:
                        break
            if answer is None:
                if stopped():
                    diagnostics.append(f"块 {bi+1}: 未检查 · 搜索提前结束")
                else:
                    diagnostics.append(f"块 {bi+1}: 未通过 · {nsym} 个纠错码字")
            else:
                changes = int(np.count_nonzero(answer != data))
                corrected[indices] = answer
                passed += 1
                diagnostics.append(f"块 {bi+1}: 通过 · 改动 {changes} 个码字 / {nsym} 个纠错码字")
        if passed > best_passed:
            best_passed = passed
            best_diagnostic = [f"{EC_NAMES[ec]} / 掩码 {mask} · {passed}/{len(blocks)} 块通过"] + diagnostics
        if passed != len(blocks):
            continue
        q = qrcode.QRCode(version=v, error_correction=ec, border=0, mask_pattern=mask)
        q.data_cache = corrected.tolist()
        q.makeImpl(False, mask)
        matrix = np.array(q.modules, np.int8)
        if np.any(grid.locked & (grid.values >= 0) & (matrix != grid.values)):
            continue
        digest = matrix.tobytes()
        if digest in seen:
            continue
        seen.add(digest)
        from .validation import validate_matrix
        validation = validate_matrix(matrix)
        if not validation.valid:
            best_diagnostic = diagnostics + ["矩阵合法性未通过："] + validation.errors
            continue
        changed = (matrix != grid.values)
        strong = changed & (confidence > .8) & (grid.values >= 0)
        # Relative image agreement; do not present as a probability of correctness.
        loss = np.where(matrix == 1, 1-grid.probability, grid.probability).mean()
        score = float(1-loss - .15*strong.mean())
        results.append(Repair(matrix, validation.text, validation.payload, EC_NAMES[ec], mask,
                              score, int(changed.sum()), int(strong.sum()), diagnostics + validation.checks))
    results.sort(key=lambda x: x.score, reverse=True)
    return results, best_diagnostic, time.monotonic() - start


def generate(data, ec="M"):
    q = qrcode.QRCode(error_correction={n: k for k, n in EC_NAMES.items()}[ec], border=0)
    q.add_data(data)
    q.make(fit=True)
    a = np.array(q.modules, np.int8)
    return Grid(a, np.where(a == 1, .99, .01).astype(np.float32), np.zeros_like(a, bool), np.zeros_like(a, np.uint8))


def repair_with_geometry(grid, image, corners, seconds=12, deep=False, inverted=False,
                         locked_corners=None, cancel=lambda: False, progress=lambda s: None):
    """Retry failed RS using at most two better-supported geometric hypotheses.

    Geometry and evidence travel with the candidate and are only applied on user
    acceptance. Manual/locked cells retain their values and provenance.
    """
    if image is None or corners is None:
        return repair(grid, seconds, deep, cancel, progress)
    from .geometry import structural_error, refine_geometry
    started = time.monotonic()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    original_error, _ = structural_error(gray, corners, grid.version, inverted)
    first_geometry, first_quality = refine_geometry(gray, corners, grid.version, inverted, cancel, locked_corners)
    can_retry = first_quality.refined and first_quality.residual < original_error-.0005
    remaining = max(0., seconds-(time.monotonic()-started))
    result, diagnostics, _ = repair(grid, remaining*(.70 if can_retry else 1.), deep, cancel, progress)
    if result or cancel():
        return result, diagnostics, time.monotonic()-started
    if not can_retry:
        return result, diagnostics, time.monotonic()-started
    manual = (grid.source == 1) | grid.locked
    best_error = original_error
    current = np.asarray(corners, np.float32)
    for iteration in range(2):
        remaining = seconds-(time.monotonic()-started)
        if cancel() or remaining < .1:
            break
        progress(f"边界反馈 {iteration+1}/2：核验结构并重新采样")
        refined, quality = ((first_geometry, first_quality) if iteration == 0 else
                            refine_geometry(gray, current, grid.version, inverted, cancel, locked_corners))
        if not quality.refined or quality.residual >= best_error-.0005:
            break
        best_error, current = quality.residual, refined
        sampled = sample_grid(image, refined, grid.version)
        if inverted:
            sampled.probability = 1-sampled.probability
            sampled.values = 1-sampled.values
        sampled = restore_structure(sampled)
        sampled.values[manual] = grid.values[manual]
        sampled.probability[manual] = grid.probability[manual]
        sampled.source[manual] = grid.source[manual]
        sampled.locked = grid.locked.copy()
        remaining = seconds-(time.monotonic()-started)
        if remaining <= 0:
            break
        found, detail, _ = repair(sampled, remaining/(2-iteration), deep, cancel, progress)
        diagnostics = detail + [f"边界结构误差 {original_error:.3f} → {quality.residual:.3f}"]
        for candidate in found:
            candidate.evidence, candidate.corners = sampled.copy(), refined.copy()
            candidate.blocks.append(f"边界反馈：结构误差 {original_error:.3f} → {quality.residual:.3f}")
        result.extend(found)
        if found:
            break
    return result, diagnostics, time.monotonic()-started
