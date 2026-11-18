"""Noninteractive distribution self-check; never touches user documents or desktop."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from .app import MainWindow, STYLE
from .engine import generate, render_matrix, repair, data_positions, read_image, locate, scan, optimize_quad, finder_mask
from .validation import validate_matrix
from .overlays import grid_colors
from .overlays import confidence_legend, contour_color
from . import __version__
from .project import save_project, load_project, export_svg


def run(output):
    report = {"ok": False, "version": __version__, "checks": []}
    start = time.monotonic()
    try:
        app = QApplication.instance() or QApplication([])
        app.setStyle("Fusion")
        app.setStyleSheet(STYLE)
        window = MainWindow()
        window.resize(1220, 710)
        window.show()
        app.processEvents()
        report["checks"].append("Qt native window created with offscreen platform")
        assert window.repair_button.height() >= 25, window.repair_button.size()
        assert window.scan_button.height() >= 25
        payload = "二维码修复 / frozen self-test".encode("utf-8")
        grid = generate(payload, "H")
        original = grid.values.copy()
        assert validate_matrix(original).valid
        for i in range(10):
            r,c = data_positions(grid.version)[i*8]
            grid.values[r,c] = -1 if i % 2 else 1-grid.values[r,c]
            grid.probability[r,c] = .49
        window.grid = grid
        assert not validate_matrix(grid.values).valid
        window.grid_view.set_grid(grid,True)
        candidates, _, _ = repair(grid, seconds=8, deep=True)
        assert candidates and candidates[0].payload == payload
        assert np.array_equal(candidates[0].matrix,original)
        report["checks"].append("Unknown/error RS correction recovered exact original matrix and bytes")
        assert validate_matrix(candidates[0].matrix).valid
        report["checks"].append("Strict structure, format, RS, segments, padding and normal/pure reader validation")
        window.repairs = candidates
        window.preview_repair(0)
        window.accept_repair()
        window.check_timer.stop()
        assert np.array_equal(window.grid.values,original)
        report["checks"].append("Candidate preview and acceptance through native window model")
        window.undo()
        window.check_timer.stop()
        assert np.any(window.grid.values < 0)
        window.redo()
        window.check_timer.stop()
        assert np.array_equal(window.grid.values,original)
        report["checks"].append("Undo/redo retained grid state")
        with tempfile.TemporaryDirectory(prefix="qrrepair-selftest-") as folder:
            path = Path(folder)
            Image.fromarray(render_matrix(original,10)).save(path/"sample.png")
            image = read_image(path/"sample.png")
            corners, contents = locate(image)
            assert corners is not None and contents[0][1] == payload
            sample = scan(image,corners,versions=[grid.version])[0]
            m = finder_mask(grid.version)
            assert (sample.grid.source[m] == 2).all()
            refined, data = optimize_quad(image,corners)
            assert refined is not None and data[0][1] == payload
            report["checks"].append("Crop-local optimization at exact QR edges and fixed initial finders")
            candidates, _, _ = repair(sample.grid,seconds=8)
            assert candidates[0].payload == payload
            report["checks"].append("PNG import, automatic localization, sampling and decoding")
            save_project(path/"sample.qrr",image,corners,window.grid)
            _,_,reloaded,_ = load_project(path/"sample.qrr")
            assert np.array_equal(reloaded.values,original)
            export_svg(path/"sample.svg",reloaded.values)
            assert (path/"sample.svg").stat().st_size > 100
            report["checks"].append("Project roundtrip and SVG export")
            window.image = image
            window.source_view.set_image(image,corners)
            window.source_view.dimension = len(window.grid.values)
            window.contour_check.setChecked(True)
            assert window.grid_view.contours is not None and window.source_view.contour_item is not None
            window.grid.source[10,10] = 1
            window.grid.probability[10,10] = .5
            assert np.array_equal(grid_colors(window.grid)[10,10],grid_colors(window.grid,True)[10,10])
            report["checks"].append("Native contour overlay and manual-cell heatmap exclusion")
            assert np.isin(sample.grid.values,[0,1]).all()
            window.grid.source[12,12] = 0
            window.grid.probability[12,12] = .5
            assert np.array_equal(grid_colors(window.grid,True)[12,12],confidence_legend()[0,0,:3])
            assert np.all(contour_color(255) < contour_color(0))
            assert window.selection_actions.isHidden()
            window.set_tool("select")
            assert not window.selection_actions.isHidden()
            window.set_tool("black")
            assert window.selection_actions.isHidden()
            window.heat_check.setChecked(True)
            assert not window.heat_legend.isHidden()
            assert window.source_view.base_item.opacity() == 1
            assert window.grid_view.contour_opacity == .65
            assert window.grid_view.base_item.opacity() == .3
            report["checks"].append("Binary sampling, red confidence legend, translucent blue contours and contextual selection controls")
            assert confidence_legend()[0,-1,3] == 0
            assert window.tabs.count() == 2 and window.colorful_background.isChecked()
            assert not window.mirror_input.isChecked() and not window.invert_input.isChecked()
            rect = window.source_view.sceneRect()
            window.source_view.focus_selection()
            assert window.source_view.sceneRect() == rect
            report["checks"].append("Rotation-only defaults, background controls, full-image selection focus and simplified light interface")
            window.sync_source_preview()
            window.hover_cell(12,12)
            assert window.pages.currentIndex() == 1
            assert window.crop_view.crop.shape == (720,720,3)
            assert window.cell_patch.pixmap().width() == 108
            assert window.contour_patch.pixmap().width() == 108
            report["checks"].append("Staged import, rectified source preview and synchronized contour detail")
            window.preview_button.setChecked(True)
            assert window.grid_view.preview_mode
            assert len(window.grid_view.scene().items()) == 1
            assert window.grid_view.base_item.pixmap().width() == (len(original)+8)*12
            assert window.grid_view.base_item.opacity() == 1
            assert window.heat_legend.isHidden()
            window.preview_button.setChecked(False)
            assert window.grid_view.base_item.opacity() == .3
            report["checks"].append("Black-white QR preview with quiet zone and return to contour editing")
            window.region_check.setChecked(True)
            assert window.grid_view.show_regions
            window.region_check.setChecked(False)
            assert not window.grid_view.show_regions
            report["checks"].append("Function-region overlay toggle")
            window.adjust_selection()
            assert window.source_view.horizontalScrollBarPolicy().name == "ScrollBarAsNeeded"
            assert window.source_view.verticalScrollBarPolicy().name == "ScrollBarAsNeeded"
            window.selection_dialog.accept()
            report["checks"].append("Selection editor scroll navigation")
        window.dirty = False
        window.close()
        app.processEvents()
        report["ok"] = True
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
    report["seconds"] = round(time.monotonic()-start,2)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0 if report["ok"] else 1
