import io
import json
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest
import qrcode

from qrrepair.engine import (Grid, generate, repair, render_matrix, decode_image, template,
    data_positions, scan, sample_grid, restore_structure, locate, oriented_corners)
from qrrepair.project import save_project, load_project, export_svg


def qr(version=3, ec=2, mask=4, payload=b"QR Repair / ground truth"):
    q = qrcode.QRCode(version=version, error_correction=ec, border=0, mask_pattern=mask)
    q.add_data(payload)
    q.make(fit=False)
    a = np.array(q.modules, np.int8)
    return Grid(a, np.where(a == 1, .98, .02).astype(np.float32), np.zeros_like(a, bool), np.zeros_like(a, np.uint8))


@pytest.mark.parametrize("version,ec,mask", [(1,1,0), (3,2,4), (7,3,7), (12,0,1), (25,2,6), (40,1,3)])
def test_exact_reconstruction(version, ec, mask):
    original = qr(version, ec, mask, b"TEST123")
    results, _, _ = repair(original, seconds=12)
    assert results
    assert any(np.array_equal(r.matrix, original.values) for r in results)
    assert results[0].payload == b"TEST123"


def test_errors_and_unknowns_with_multiblock_correction():
    original = qr(7, 2, 5)
    damaged = original.copy()
    positions = data_positions(7)
    # Spread erasures and bit errors across interleaved blocks.
    for i in range(24):
        r, c = positions[i*8]
        damaged.values[r,c] = -1 if i % 2 else 1-damaged.values[r,c]
        damaged.probability[r,c] = .51 if damaged.values[r,c] == 1 else .49
    results, diagnostics, _ = repair(damaged, seconds=10, deep=True)
    assert results, diagnostics
    assert np.array_equal(results[0].matrix, original.values)


def test_lock_is_never_overridden():
    g = qr()
    r, c = data_positions(3)[0]
    g.values[r,c] ^= 1
    g.locked[r,c] = True
    results, _, _ = repair(g, seconds=3)
    assert all(r.matrix[g.locked].tolist() == g.values[g.locked].tolist() for r in results)
    assert not results


def test_structure_preserves_locks_and_evidence():
    g = qr()
    g.values[0,0] = 0
    g.locked[0,0] = True
    g.values[0,1] = -1
    out = restore_structure(g)
    assert out.values[0,0] == 0
    assert out.values[0,1] == 1
    assert np.array_equal(out.probability, g.probability)


def test_perspective_blur_version_selection_and_recovery():
    g = qr(4, 2, 2)
    n = len(g.values)
    clean = render_matrix(g.values, 12, border=0)
    source = np.repeat(clean[:,:,None], 3, 2)
    src = np.float32([[0,0],[n*12,0],[n*12,n*12],[0,n*12]])
    corners = np.float32([[90,65],[550,115],[505,590],[45,485]])
    h = cv2.getPerspectiveTransform(src, corners)
    image = cv2.warpPerspective(source, h, (650,650), borderValue=(255,255,255))
    image = cv2.GaussianBlur(image, (0,0), 2.6)
    samples = scan(image, corners, versions=range(1,9))
    assert samples[0].grid.version == 4
    results, _, _ = repair(samples[0].grid, seconds=10)
    assert results
    assert np.array_equal(results[0].matrix, g.values)


def test_rotated_mirrored_negative_candidate():
    g = qr(3,2,3)
    a = 1-np.rot90(np.fliplr(g.values))
    gray = render_matrix(a, 10, border=0)
    image = np.repeat(gray[:,:,None], 3, 2)
    n = image.shape[0]
    corners = np.float32([[0,0],[n,0],[n,n],[0,n]])
    candidate = scan(image, corners, versions=[3], mirror=True, invert=True)[0]
    assert candidate.inverted
    oriented = oriented_corners(corners, candidate)
    resampled = sample_grid(image, oriented, 3)
    prob = 1-resampled.probability
    assert np.mean((prob>.5) == candidate.grid.values) > .99
    results, _, _ = repair(candidate.grid, seconds=8)
    assert results
    assert results[0].payload == b"QR Repair / ground truth"


def test_project_roundtrip_and_export(tmp_path):
    g = qr()
    g.values[12,12] = -1
    g.locked[2,2] = True
    g.source[12,12] = 1
    image = np.full((100,100,3), 255, np.uint8)
    corners = np.float32([[0,0],[100,0],[100,100],[0,100]])
    path = tmp_path / "中文工程.qrr"
    save_project(path, image, corners, g, {"test": 1})
    actual, pts, restored, settings = load_project(path)
    assert np.array_equal(image, actual)
    assert np.array_equal(corners, pts)
    for name in ("values","probability","locked","source"):
        assert np.array_equal(getattr(g,name),getattr(restored,name))
    assert settings == {"test":1}
    with pytest.raises(ValueError):
        export_svg(tmp_path/"unknown.svg",g.values)


def test_cancel_and_bad_geometry():
    g = qr()
    results, _, elapsed = repair(g, cancel=lambda: True)
    assert results == [] and elapsed < 1
    with pytest.raises(ValueError):
        scan(np.zeros((100,100,3),np.uint8), [[0,0],[100,100],[100,0],[0,100]])


def test_unicode_binary_payload_and_detection():
    data = "模糊二维码修复测试".encode("utf-8") + b"\x00\xff"
    g = generate(data, "H")
    gray = render_matrix(g.values)
    result = decode_image(gray)
    assert result and bytes(result[0].bytes) == data
    corners, contents = locate(np.repeat(gray[:,:,None],3,2))
    assert corners is not None and contents[0][1] == data
