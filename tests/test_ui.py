import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt, QPoint
from qrrepair.app import MainWindow
from qrrepair.engine import generate, repair


def test_manual_three_finder_initialization_and_region_reset():
    import numpy as np
    from qrrepair.engine import render_matrix
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    g = generate(b"manual finder control")
    gray = render_matrix(g.values,10,border=0)
    w.image = np.repeat(gray[:,:,None],3,2)
    w.reset_corners()
    w.finder_version = g.version
    w.run_job = lambda fn,done:done(fn(lambda:False,lambda message:None))
    n = len(g.values)
    w.finders_picked(np.float32([[35,35],[(n-3.5)*10,35],[35,(n-3.5)*10]]))
    assert np.array_equal(w.grid.values,g.values)
    w.boundary_candidates = [('stale',)]
    w.corner_locks[0].setChecked(True)
    w.reset_document()
    assert w.boundary_candidates==[] and not w.corner_locks[0].isChecked()
    w.dirty=False
    w.close()


def test_edit_undo_redo_preview_and_accept():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    g = generate(b"Desktop editor integration test", "H")
    original = g.values.copy()
    window.grid = g
    window.grid_view.set_grid(g, True)
    window.push_history()
    window.grid.values[10,10] ^= 1
    window.edited()
    window.check_timer.stop()
    window.undo()
    window.check_timer.stop()
    assert np.array_equal(window.grid.values, original)
    window.redo()
    window.check_timer.stop()
    assert window.grid.values[10,10] != original[10,10]
    results, _, _ = repair(window.grid, seconds=5)
    assert results
    window.repairs = results
    window.preview_repair(0)
    assert window.grid_view.preview is not None
    assert window.grid.values[10,10] != original[10,10]
    window.accept_repair()
    window.check_timer.stop()
    assert np.array_equal(window.grid.values, original)
    assert window.payload == b"Desktop editor integration test"
    window.dirty = False
    window.close()
    app.processEvents()


def test_finders_protected_and_crop_overlay_tracks_geometry():
    from qrrepair.engine import render_matrix, finder_mask
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    g = generate(b"Overlay native test")
    window.grid = g
    window.grid_view.set_grid(g,True)
    image = np.repeat(render_matrix(g.values,10,border=0)[:,:,None],3,2)
    n = len(image)
    p = np.float32([[0,0],[n,0],[n,n],[0,n]])
    window.image = image
    window.source_view.set_image(image,p)
    window.source_view.dimension = len(g.values)
    window.contour_check.setChecked(True)
    assert window.grid_view.contours is not None
    assert window.source_view.contour_item is not None
    mapped = window.source_view.contour_item.mapToScene(720,720)
    assert abs(mapped.x()-n) < .01 and abs(mapped.y()-n) < .01
    original = g.values.copy()
    window.grid_view.tool = "white"
    window.grid_view.paint_cell((0,0))
    assert g.values[0,0] == original[0,0]
    window.grid_view.selection = (0,0,10,10)
    window.fill_selection(0)
    window.check_timer.stop()
    mask = finder_mask(g.version)
    assert np.array_equal(g.values[mask],original[mask])
    assert g.source[9,9] == 1 and g.values[9,9] == 0
    assert window.selection_actions.isHidden()
    window.set_tool("select")
    assert not window.selection_actions.isHidden()
    window.set_tool("white")
    assert window.selection_actions.isHidden()
    assert window.heat_legend.isHidden()
    window.heat_check.setChecked(True)
    assert not window.heat_legend.isHidden()
    assert window.source_view.base_item.opacity() == 1
    assert window.grid_view.base_item.opacity() == .3
    assert window.grid_view.contour_opacity == .65
    window.contour_check.setChecked(False)
    assert window.source_view.base_item.opacity() == 1
    assert window.grid_view.base_item.opacity() == 1
    window.contour_check.setChecked(True)
    window.corners_changed(p)
    assert window.grid_view.contours is None  # No stale overlay on old sampled grid.
    assert window.source_view.contour_item is not None  # Crop evidence still available.
    window.dirty = False
    window.close()
    app.processEvents()


def test_focus_preserves_full_image_handles_and_background_settings(tmp_path):
    from PySide6.QtCore import QPointF
    from qrrepair.project import load_project
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    assert w.tabs.count() == 2
    assert not w.mirror_input.isChecked() and not w.invert_input.isChecked()
    assert w.colorful_background.isChecked()
    w.image = np.full((800,1000,3),220,np.uint8)
    p = np.float32([[300,250],[650,250],[650,600],[300,600]])
    w.source_view.set_image(w.image,p)
    w.adjust_selection()
    app.processEvents()
    w.source_view.fit()
    full = w.source_view.transform().m11()
    w.source_view.focus_selection()
    assert w.source_view.transform().m11() > full
    assert w.source_view.sceneRect().width() == 1000
    w.source_view.handles[0].setPos(QPointF(200,200))
    assert w.source_view.corners[0].tolist() == [200,200]
    w.source_view.fit()
    assert abs(w.source_view.transform().m11()-full) < .001
    w.selection_dialog.accept()
    w.set_background_color((220,220,220))
    assert w.background_button.text() == '#DCDCDC'
    w.project_path = str(tmp_path/'settings.qrr')
    w.mirror_input.setChecked(True)
    w.invert_input.setChecked(True)
    w.colorful_background.setChecked(False)
    assert w.save()
    _,_,_,settings = load_project(w.project_path)
    assert settings['background_color'] == [220,220,220]
    assert settings['mirror_input'] and settings['invert_input']
    w.set_background_color(None)
    w.dirty = False
    w.open_project(w.project_path)
    w.check_timer.stop()
    assert w.background_color == (220,220,220)
    assert not w.colorful_background.isChecked()
    w.dirty = False
    w.close()
    app.processEvents()


def test_staged_import_rectifies_rotated_selection_and_fits_on_resize():
    from qrrepair.engine import render_matrix, sample_grid
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.resize(1300,850)
    w.show()
    app.processEvents()
    assert w.pages.currentIndex() == 0
    assert not w.splitter.isVisible()
    w.welcome_mirror.setChecked(True)
    assert w.mirror_input.isChecked()
    w.welcome_mirror.setChecked(False)
    g = generate(b"upright crop test", "H")
    # Run the worker function inline to verify exactly the import orientation path.
    w.run_job = lambda fn, done: done(fn(lambda:False,lambda message:None))
    for rotation in range(4):
        gray = np.rot90(render_matrix(g.values,10),rotation).copy()
        w.image = np.repeat(gray[:,:,None],3,2)
        w.reset_corners()
        w.auto_locate()
        app.processEvents()
        assert w.pages.currentIndex() == 1
        assert w.crop_view.crop.shape == (720,720,3)
        recovered = sample_grid(w.image,w.source_view.corners,g.version)
        assert np.array_equal(recovered.values,g.values)
        assert w.crop_view.sceneRect().width() == 720
        visible = w.crop_view.mapFromScene(w.crop_view.sceneRect()).boundingRect()
        assert w.crop_view.viewport().rect().contains(visible)
    before = w.crop_view.transform().m11()
    w.resize(1080,660)
    app.processEvents()
    assert w.crop_view.transform().m11() < before
    visible = w.crop_view.mapFromScene(w.crop_view.sceneRect()).boundingRect()
    assert w.crop_view.viewport().rect().contains(visible)
    w.dirty = False
    w.close()
    app.processEvents()


def test_detail_contour_matches_canvas_and_tracks_precision():
    from qrrepair.engine import render_matrix
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    g = generate(b"detail")
    g.source[:] = 0
    g.probability[:] = .5
    g.source[0,1] = 1
    w.grid = g
    w.image = np.repeat(render_matrix(g.values,10,border=0)[:,:,None],3,2)
    size = len(w.image)
    w.source_view.set_image(w.image,np.float32([[0,0],[size,0],[size,size],[0,size]]))
    w.source_view.dimension = len(g.values)
    w.update_contours()
    w.hover_cell(0,0)
    img = w.contour_patch.pixmap().toImage()
    assert img.pixelColor(10,10).getRgb() != img.pixelColor(50,50).getRgb()
    assert w.cell_patch.pixmap().size() == w.contour_patch.pixmap().size()
    # The detail uses source evidence, including manually edited modules.
    first = w.contour_patch.pixmap().toImage()
    g.values[0,0] ^= 1
    g.source[0,0] = 1
    w.hover_cell(0,0)
    assert first == w.contour_patch.pixmap().toImage()
    w.image[:] = 100
    w.contour_key = None
    w.update_contours()
    first = w.contour_patch.pixmap().toImage()
    w.contour_interval.setValue(64)
    assert first != w.contour_patch.pixmap().toImage()
    assert w.crop_view.highlight == (0,0)
    w.corners_changed(w.source_view.corners)
    assert w.cell_patch.pixmap().isNull() and w.contour_patch.pixmap().isNull()
    w.dirty = False
    w.close()
    app.processEvents()


def test_adjust_selection_enables_scroll_navigation():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.image = np.zeros((2400, 3200, 3), np.uint8)
    w.source_view.set_image(w.image, np.float32([[200,200],[3000,200],[3000,2200],[200,2200]]))
    w.adjust_selection()
    assert w.source_view.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert w.source_view.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    w.selection_dialog.accept()
    w.dirty = False
    w.close()
    app.processEvents()


def test_preview_shows_exact_matrix_with_quiet_zone_and_blocks_paint():
    from qrrepair.engine import render_matrix, decode_image
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.grid = generate(b"preview roundtrip", "H")
    w.grid_view.set_grid(w.grid, True)
    w.update_info()
    w.heat_check.setChecked(True)
    w.grid_view.contours = np.zeros((720,720,4),np.uint8)
    w.contour_check.setChecked(True)
    w.preview_button.setChecked(True)
    assert w.grid_view.preview_mode
    assert w.heat_legend.isHidden() and w.brush_actions.isHidden()
    assert len(w.grid_view.scene().items()) == 1
    def pixels():
        image = w.grid_view.base_item.pixmap().toImage()
        return np.array([[image.pixelColor(x,y).red() for x in range(image.width())]
                         for y in range(image.height())],np.uint8)
    rendered = pixels()
    assert np.array_equal(rendered,render_matrix(w.grid.values,12,4))
    assert decode_image(rendered)[0].bytes == b"preview roundtrip"
    original = w.grid.values.copy()
    w.grid_view.paint_cell((10,10))
    assert np.array_equal(w.grid.values,original)
    candidate = generate(b"candidate preview", "H").values
    w.grid_view.preview = candidate
    w.grid_view.refresh()
    assert np.array_equal(pixels(),render_matrix(candidate,12,4))
    w.preview_button.setChecked(False)
    assert not w.grid_view.preview_mode
    assert not w.heat_legend.isHidden() and not w.brush_actions.isHidden()
    assert np.array_equal(w.grid.values,original)
    w.dirty = False
    w.close()
    app.processEvents()


def test_region_overlay_toggle_marks_qr_function_areas():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.grid = generate(b"regions")
    w.grid_view.set_grid(w.grid, True)
    assert not w.grid_view.show_regions
    w.region_check.setChecked(True)
    assert w.grid_view.show_regions
    before = len(w.grid_view.scene().items())
    w.grid_view.viewport().update()
    assert before >= 1
    w.region_check.setChecked(False)
    assert not w.grid_view.show_regions
    w.dirty = False
    w.close()
    app.processEvents()
