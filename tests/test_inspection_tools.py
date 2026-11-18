import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication
from qrrepair.app import MainWindow
from qrrepair.engine import generate, region_masks
from qrrepair.inspection import inspect_module


def test_tools_cycle_and_inspect_without_editing():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.grid = generate(b"inspect", "H")
    v = w.grid_view
    v.set_grid(w.grid)
    cell = tuple(np.argwhere(region_masks(w.grid.version)["数据"])[0])
    v.tool = "cycle"
    for expected in [-1,1,0,-1]:
        v.visited.clear()
        v.paint_cell(cell)
        assert w.grid.values[cell] == expected
    v.tool = "invert"
    v.visited.clear()
    v.paint_cell(cell)
    assert w.grid.values[cell] == -1
    w.grid = generate(b"inspect", "H")
    v.set_grid(w.grid)
    before = w.grid.values.copy()
    v.tool = "inspect_block"
    w.inspect_cell(8,0)
    assert "30%" in w.inspection.text()
    assert np.array_equal(before,w.grid.values)
    assert v.inspected_mask.any()
    w.close()


def test_function_regions_and_data_readout():
    g = generate(b"read data", "H")
    mask,text = inspect_module(g.values,7,0)
    assert "分隔" in text
    assert mask.sum()==15
    cell = tuple(np.argwhere(region_masks(g.version)["数据"])[0])
    mask,text = inspect_module(g.values,*cell)
    assert "RS 块" in text and "0x" in text
