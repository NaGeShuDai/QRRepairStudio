import cv2
import numpy as np

from qrrepair.engine import (generate, render_matrix, data_positions, locate,
                             locate_candidates, scan, sample_grid, oriented_corners)
from qrrepair.geometry import refine_geometry


def rgb(a):
    return np.repeat(a[:, :, None], 3, 2)


def test_undecodable_symbol_retains_boundary_and_no_fake_payload():
    g = generate(b"Damaged data, intact geometric evidence", "H")
    a = g.values.copy()
    positions = np.array(data_positions(g.version))
    indices = np.random.default_rng(21).choice(len(positions), len(positions)//3, replace=False)
    a[positions[indices, 0], positions[indices, 1]] ^= 1
    image = rgb(render_matrix(a, 8))
    p, contents = locate(image)
    assert p is not None and contents == []
    end = (len(a)+4)*8
    assert np.max(np.abs(p-np.float32([[32,32],[end,32],[end,end],[32,end]]))) < 3


def test_multiple_symbols_remain_separate():
    codes = [rgb(render_matrix(generate(s).values, 8)) for s in (b"LEFT", b"RIGHT")]
    image = np.full((400, 850, 3), 255, np.uint8)
    for a, x in zip(codes, (25, 500)):
        image[40:40+len(a), x:x+len(a)] = a
    found = locate_candidates(image)
    assert len(found) == 2
    assert {d.payload for d in found} == {b"LEFT", b"RIGHT"}


def test_blank_and_cancelled_detection_do_not_create_regions():
    image = np.full((240, 300, 3), 180, np.uint8)
    assert locate_candidates(image) == []
    assert locate_candidates(image, cancel=lambda: True) == []


def test_missing_pixels_never_become_confident_white():
    g = generate(b"coverage")
    image = rgb(render_matrix(g.values, 10, border=0))
    n = len(image)
    p = np.float32([[-50,0],[n-50,0],[n-50,n],[-50,n]])
    sampled = sample_grid(image, p, g.version)
    assert np.all(sampled.probability[:, :4] == .5)


def test_refinement_reduces_grid_error_and_respects_corner_locks():
    g = generate(b"geometric registration target / 2026", "H")
    n = len(g.values)
    truth = np.float32([[70,45],[445,75],[410,450],[35,410]])
    src = np.float32([[0,0],[n*10,0],[n*10,n*10],[0,n*10]])
    image = cv2.warpPerspective(render_matrix(g.values,10,border=0),
        cv2.getPerspectiveTransform(src, truth), (500,500), borderValue=255)
    image = cv2.GaussianBlur(image,(0,0),1.2)
    wrong = truth+np.float32([[5,4],[-4,5],[5,-4],[-4,3]])
    fixed, quality = refine_geometry(image, wrong, g.version)
    assert quality.refined
    assert np.linalg.norm(fixed-truth,axis=1).mean() < np.linalg.norm(wrong-truth,axis=1).mean()*.7
    locked, _ = refine_geometry(image, wrong, g.version, locked=[True,False,False,False])
    assert np.array_equal(locked[0], wrong[0])


def test_refined_sample_geometry_matches_its_grid():
    g = generate(b"orientation and geometry", "H")
    image = rgb(np.rot90(render_matrix(g.values, 10)))
    p, _ = locate(image)
    best = scan(image, p, versions=[g.version])[0]
    sampled = sample_grid(image, oriented_corners(p,best), g.version)
    assert np.array_equal(sampled.values, g.values)
    assert best.geometry.coverage > .99


def test_feedback_preserves_manual_evidence_and_carries_geometry(monkeypatch):
    from qrrepair import engine, geometry
    g = generate(b"feedback")
    image = rgb(render_matrix(g.values,10,border=0))
    n = len(image)
    corners = np.float32([[0,0],[n,0],[n,n],[0,n]])
    refined = corners+np.float32([.1,.1])
    r,c = data_positions(g.version)[0]
    g.source[r,c], g.locked[r,c] = 1, True
    original = g.copy()
    monkeypatch.setattr(geometry,'structural_error',lambda *a,**k:(.2,1.))
    monkeypatch.setattr(geometry,'refine_geometry',lambda *a,**k:(refined,geometry.GeometryQuality(.1,1.,.1,True)))
    calls = []
    def fake_repair(grid, *args):
        calls.append(grid.copy())
        if len(calls)==1:
            return [],[],0.
        return [engine.Repair(grid.values.copy(),'feedback',b'feedback','M',0,1.,0,0,[])],[],0.
    monkeypatch.setattr(engine,'repair',fake_repair)
    results,_,_ = engine.repair_with_geometry(g,image,corners,seconds=2)
    assert results and len(calls)==2
    assert calls[1].values[r,c]==original.values[r,c] and calls[1].locked[r,c]
    assert calls[1].probability[r,c]==original.probability[r,c]
    assert np.array_equal(results[0].corners,refined)
    assert np.array_equal(g.values,original.values)


def test_candidate_ranking_rejects_spurious_inner_symbol():
    from tools.benchmark_boundary import make_case
    image,truth,matrix,version,_ = make_case(8)
    found = locate_candidates(image)
    assert len(found)==1
    best = scan(image,found[0].corners)[0]
    assert best.grid.version==version
    assert np.array_equal(best.grid.values,matrix)
