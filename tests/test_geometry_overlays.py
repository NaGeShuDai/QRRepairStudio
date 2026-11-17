import cv2
import numpy as np
from qrrepair.engine import generate, render_matrix, optimize_quad, scan, finder_mask, template, warp
from qrrepair.overlays import grayscale_contours, grid_colors, contour_color, confidence_legend, confidence_colors


def rgb(a):
    return np.repeat(a[:,:,None],3,2)


def test_optimization_stays_inside_selected_qr_not_other_symbol():
    a = rgb(render_matrix(generate(b"LEFT").values,8))
    b = rgb(render_matrix(generate(b"RIGHT").values,8))
    canvas = np.full((400,850,3),255,np.uint8)
    canvas[40:40+len(a),25:25+len(a)] = a
    canvas[60:60+len(b),500:500+len(b)] = b
    roi = np.float32([[490,50],[500+len(b)+10,50],[500+len(b)+10,70+len(b)],[490,70+len(b)]])
    found, contents = optimize_quad(canvas,roi)
    assert found is not None and contents[0][1] == b"RIGHT"
    assert (found[:,0] > 500).all()


def test_exact_edges_and_outside_image_padding():
    g = generate(b"boundary")
    a = rgb(render_matrix(g.values,10,border=0))
    n = len(a)
    for d in (0,12):
        p = np.float32([[-d,-d],[n+d,-d],[n+d,n+d],[-d,n+d]])
        found, contents = optimize_quad(a,p)
        assert found is not None
        assert np.all(found >= 0) and np.all(found <= n)
        assert np.max(np.abs(found - np.float32([[0,0],[n,0],[n,n],[0,n]]))) < 3
        assert contents[0][1] == b"boundary"
    cropped = warp(a,np.float32([[-50,-50],[n,-50],[n,n],[-50,n]]),500)
    assert cropped[0,0].tolist() == [255,255,255]


def test_failure_never_changes_selection_or_searches_outside():
    g = generate(b"outside")
    a = rgb(render_matrix(g.values,8))
    canvas = np.full((len(a),len(a)*2,3),255,np.uint8)
    canvas[:,:len(a)] = a
    n = len(a)
    p = np.float32([[n+5,5],[n*2-5,5],[n*2-5,n-5],[n+5,n-5]])
    old = p.copy()
    found, _ = optimize_quad(canvas,p)
    assert found is None and np.array_equal(p,old)


def test_initial_scan_has_fixed_finders_despite_damaged_evidence():
    g = generate(b"finder first", "H")
    a = g.values.copy()
    a[0:3,0:3] = 0
    image = rgb(render_matrix(a,10,border=0))
    n = len(image)
    s = scan(image,np.float32([[0,0],[n,0],[n,n],[0,n]]),versions=[g.version])[0]
    m = finder_mask(g.version)
    assert np.array_equal(s.grid.values[m],template(g.version)[m])
    assert (s.grid.source[m] == 2).all()
    assert (s.grid.probability[:3,:3] < .5).all()  # Evidence is not rewritten.


def test_contour_precision_and_manual_heatmap_exclusion():
    gradient = np.tile(np.arange(256,dtype=np.uint8),(256,1))
    fine, coarse = grayscale_contours(gradient,8), grayscale_contours(gradient,64)
    assert (fine[:,:,3] == 255).all()
    assert len(np.unique(fine[:,:,:3].reshape(-1,3),axis=0)) > len(np.unique(coarse[:,:,:3].reshape(-1,3),axis=0))*2
    assert not np.array_equal(fine[:,:,0],fine[:,:,1])
    palette = contour_color(np.arange(256))
    assert np.all(np.diff(palette.astype(int),axis=0) <= 0)
    assert np.all(palette[:,2] >= palette[:,1])
    assert palette[0].tolist() == [255,255,255]
    assert palette[-1].tolist() == [12,45,105]
    g = generate(b"heat")
    g.probability[:] = .5
    g.source[10,10] = g.source[10,11] = 1
    g.values[10,11] = -1
    plain, heat = grid_colors(g),grid_colors(g,True)
    assert np.array_equal(plain[10,10:12],heat[10,10:12])
    assert not np.array_equal(plain[12,12],heat[12,12])
    assert np.array_equal(heat[12,12],confidence_legend()[0,0,:3])
    g.probability[12,12] = 1
    assert np.array_equal(grid_colors(g,True)[12,12],grid_colors(g,False)[12,12])


def test_low_confidence_sampling_is_binary_and_legacy_unknowns_resolve():
    from qrrepair.engine import resolve_unknowns, sample_grid
    im = np.full((210,210,3),128,np.uint8)
    p = np.float32([[0,0],[210,0],[210,210],[0,210]])
    g = sample_grid(im,p,1,uncertain=.95)
    assert np.isin(g.values,[0,1]).all()
    assert np.isin(scan(im,p,versions=[1],uncertain=.95)[0].grid.values,[0,1]).all()
    g.values[10,10:13] = -1
    g.probability[10,10:13] = [.49,.51,.5]
    g.source[10,10:13] = 1
    out = resolve_unknowns(g)
    assert out.values[10,10:13].tolist() == [0,1,1]
    assert (out.source[10,10:13] == 0).all()
    assert (g.values[10,10:13] == -1).all()


def test_red_heat_alpha_and_explicit_transform_options():
    rgba = confidence_colors(np.linspace(0,1,11))
    assert (rgba[:,:3] == [235,45,35]).all()
    assert rgba[0,3] == 255 and rgba[-1,3] == 0
    assert (np.diff(rgba[:,3].astype(int)) < 0).all()
    g = generate(b"Explicit mirror and inversion", "H")
    a = rgb(render_matrix(1-np.fliplr(g.values),10,border=0))
    n = len(a)
    corners = np.float32([[0,0],[n,0],[n,n],[0,n]])
    default = scan(a,corners,versions=[g.version])[0]
    assert not default.mirrored and not default.inverted
    chosen = scan(a,corners,versions=[g.version],mirror=True,invert=True)[0]
    assert chosen.mirrored and chosen.inverted
    assert np.array_equal(chosen.grid.values,g.values)


def test_background_options_detection_and_original_preserved():
    from qrrepair.engine import background_mask, detection_image
    a = np.array([[[255,0,0],[0,0,0],[128,128,128],[255,255,255]]],np.uint8)
    assert background_mask(a).tolist() == [[True,False,False,False]]
    assert background_mask(a,(128,128,128),False,10).tolist() == [[False,False,True,False]]
    filtered = detection_image(a,colorful=True)
    assert filtered[0,0].tolist() == [255,255,255]
    assert a[0,0].tolist() == [255,0,0]
    assert np.array_equal(detection_image(a,colorful=False),a)
    g = generate(b"Colored background")
    qr = rgb(render_matrix(g.values,8,border=0))
    n = len(qr)
    image = np.full((n+80,n+80,3),[180,25,35],np.uint8)
    image[40:40+n,40:40+n] = qr
    p = np.float32([[0,0],[n+80,0],[n+80,n+80],[0,n+80]])
    corners, data = optimize_quad(image,p,colorful=True)
    assert corners is not None and data[0][1] == b"Colored background"
    assert np.max(np.abs(corners-np.float32([[40,40],[n+40,40],[n+40,n+40],[40,n+40]]))) < 3
