from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import traceback

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QColor, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget,
    QPlainTextEdit, QSplitter, QFileDialog, QMessageBox, QToolBar, QButtonGroup,
    QDialog, QDialogButtonBox, QFormLayout, QTabWidget, QGroupBox, QProgressBar,
    QScrollArea, QLayout, QColorDialog, QGraphicsDropShadowEffect, QStackedWidget, QInputDialog, QStyle, QToolButton, QMenu)

from . import __version__
from .engine import (Grid, read_image, locate, scan, repair, render_matrix, generate,
                     restore_structure, oriented_corners, decode_image, fixed_mask,
                     finder_mask, optimize_quad, warp, resolve_unknowns)
from .engine import locate_candidates, repair_with_geometry
from .validation import validate_matrix, require_valid
from .overlays import grayscale_contours, confidence_legend
from .project import save_project, load_project, export_svg
from .canvas import SourceView, CropView, GridView, pixmap

from .theme import STYLE


class Worker(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.stop = threading.Event()

    def run(self):
        try:
            self.result.emit(self.fn(self.stop.is_set, self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


def button(text, callback, primary=False):
    b = QPushButton(text)
    b.clicked.connect(callback)
    if primary:
        b.setObjectName("primary")
    return b


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR Repair Studio · 二维码修复工作台")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width()-40), min(960, screen.height()-60))
        self.setMinimumSize(1080, 660)
        self.image = None
        self.grid = None
        self.project_path = None
        self.dirty = False
        self.history = []
        self.future = []
        self.samples = []
        self.repairs = []
        self.scan_corners = None
        self.selected_repair = None
        self.payload = None
        self.worker = None
        self.pending_verify = False
        self.inverted = False
        self.contour_key = None
        self.contour_layer = None
        self.background_color = None
        self.binary_mode = False
        self.inspect_mode = "single"
        self.scan_geometry_mirrored = False
        self.boundary_candidates = []
        self.geometry_status = ""
        self.check_timer = QTimer(self)
        self.check_timer.setSingleShot(True)
        self.check_timer.setInterval(650)
        self.check_timer.timeout.connect(self.quick_verify)
        self.setAcceptDrops(True)
        self.build_ui()
        self.update_info()

    def action(self, title, fn, shortcut=None):
        a = QAction(title, self)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        return a

    def build_ui(self):
        self.toolbar = QToolBar("文件与编辑")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        for text, fn, key in [("导入图片", self.open_image, "Ctrl+O"), ("粘贴", self.paste_image, "Ctrl+V"),
                              ("打开工程", self.open_project, "Ctrl+Shift+O"), ("保存工程", self.save, "Ctrl+S"),
                              ("另存工程", lambda: self.save(True), "Ctrl+Shift+S"),
                              ("生成", self.generate_dialog, "Ctrl+N")]:
            self.toolbar.addAction(self.action(text, fn, key))
        self.toolbar.addSeparator()
        self.undo_action = self.action("撤销", self.undo, "Ctrl+Z")
        self.redo_action = self.action("重做", self.redo, "Ctrl+Y")
        self.toolbar.addActions([self.undo_action, self.redo_action])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action("导出", self.export_dialog, "Ctrl+E"))
        self.toolbar.addAction(self.action("使用说明", self.help_dialog, "F1"))

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        head = QHBoxLayout()
        title = QLabel("QR REPAIR  /  二维码修复工作台")
        title.setObjectName("title")
        head.addWidget(title)
        head.addStretch()
        self.summary = QLabel("")
        self.summary.setObjectName("muted")
        head.addWidget(self.summary)
        root.addLayout(head)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        welcome = QWidget()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.addStretch()
        card = QWidget()
        card.setMinimumWidth(320)
        card.setMaximumWidth(400)
        card_layout = QVBoxLayout(card)
        title = QLabel("修复二维码")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        card_layout.addWidget(button("导入图片", self.open_image, True))
        row = QHBoxLayout()
        row.addWidget(button("粘贴图片", self.paste_image))
        row.addWidget(button("打开工程", self.open_project))
        card_layout.addLayout(row)
        welcome_options = QHBoxLayout()
        self.welcome_mirror = QCheckBox("镜像")
        self.welcome_invert = QCheckBox("反色")
        welcome_options.addWidget(self.welcome_mirror)
        welcome_options.addWidget(self.welcome_invert)
        card_layout.addLayout(welcome_options)
        welcome_layout.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        welcome_layout.addStretch()
        self.pages.addWidget(welcome)
        workspace = QWidget()
        self.pages.addWidget(workspace)
        root = QVBoxLayout(workspace)
        root.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.splitter, 1)

        self.left = QWidget()
        left = QVBoxLayout(self.left)
        left.setContentsMargins(0, 0, 9, 0)
        label = QLabel("选区原图")
        label.setObjectName("title")
        left.addWidget(label)
        options = QHBoxLayout()
        self.mirror_input = QCheckBox("镜像")
        self.invert_input = QCheckBox("反色")
        self.mirror_input.setToolTip("按镜像图片采样")
        self.invert_input.setToolTip("按反色图片采样")
        options.addWidget(self.mirror_input)
        options.addWidget(self.invert_input)
        options.addStretch()
        self.welcome_mirror.toggled.connect(self.mirror_input.setChecked)
        self.mirror_input.toggled.connect(self.welcome_mirror.setChecked)
        self.welcome_invert.toggled.connect(self.invert_input.setChecked)
        self.invert_input.toggled.connect(self.welcome_invert.setChecked)
        self.source_view = SourceView()
        self.source_view.cornersChanged.connect(self.corners_changed)
        self.source_view.colorPicked.connect(self.set_background_color)
        self.source_view.findersPicked.connect(self.finders_picked)
        self.source_view.set_image(None)
        self.selection_dialog = QDialog(self)
        self.selection_dialog.setWindowTitle("调整选区")
        self.selection_dialog.resize(900, 760)
        selection_layout = QVBoxLayout(self.selection_dialog)
        selection_layout.addWidget(self.source_view, 1)
        row = QHBoxLayout()
        row.addWidget(button("优化框选", lambda: self.auto_locate(True)))
        row.addWidget(button("全图定位", self.auto_locate))
        row.addWidget(button("重置四角", self.reset_corners))
        row.addWidget(button("三点定位", self.pick_finders))
        row.addWidget(button("聚焦", self.source_view.focus_selection))
        row.addWidget(button("全图", self.source_view.fit))
        row.addWidget(button("完成", self.selection_dialog.accept, True))
        selection_layout.addLayout(row)
        locks = QHBoxLayout()
        locks.addWidget(QLabel("固定角点"))
        self.corner_locks = []
        for name in ("左上", "右上", "右下", "左下"):
            check = QCheckBox(name)
            self.corner_locks.append(check)
            locks.addWidget(check)
        selection_layout.addLayout(locks)
        self.region_list = QComboBox()
        self.region_list.setToolTip("多码图片可在这里选择目标二维码。")
        self.region_list.currentIndexChanged.connect(self.choose_boundary)
        selection_layout.addWidget(self.region_list)
        self.geometry_label = QLabel("")
        self.geometry_label.setWordWrap(True)
        selection_layout.addWidget(self.geometry_label)
        self.crop_view = CropView()
        left.addWidget(self.crop_view, 2)
        self.corner_info = QLabel("")
        self.corner_info.setObjectName("muted")
        left.addWidget(self.corner_info)
        row = QHBoxLayout()
        row.addWidget(button("调整选区", self.adjust_selection))
        row.addWidget(button("优化框选", lambda: self.auto_locate(True)))
        row.addWidget(button("旋转", self.rotate_crop))
        left.addLayout(row)
        self.options_panel = QWidget()
        controls = QVBoxLayout(self.options_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addLayout(options)
        sampling = QGroupBox("网格采样")
        form = QFormLayout(sampling)
        self.version = QComboBox()
        self.version.addItem("自动搜索版本 1–40", 0)
        for i in range(1, 41):
            self.version.addItem(f"V{i}  ·  {17+4*i} × {17+4*i}", i)
        form.addRow("矩阵尺寸", self.version)
        self.bias = QSpinBox()
        self.bias.setRange(-80, 80)
        self.bias.setSuffix("  灰度")
        self.bias.setToolTip("正值倾向黑色，负值倾向白色；调整后重新采样。")
        form.addRow("阈值偏移", self.bias)
        self.scan_button = button("生成网格", self.start_scan, True)
        form.addRow(self.scan_button)
        self.binary_button = button("二值化处理", self.toggle_binary)
        form.addRow(self.binary_button)
        controls.addWidget(sampling)
        options_scroll = QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        options_scroll.setWidget(self.options_panel)
        options_scroll.setMinimumHeight(230)
        left.addWidget(options_scroll, 3)
        self.left.setMinimumWidth(280)
        self.splitter.addWidget(self.left)

        self.middle = QWidget()
        mid = QVBoxLayout(self.middle)
        mid.setContentsMargins(9, 0, 9, 0)
        label = QLabel("二维码")
        label.setObjectName("title")
        view_row = QHBoxLayout()
        view_row.addWidget(label)
        view_row.addStretch()
        self.preview_button = QPushButton("预览视图")
        self.preview_button.setCheckable(True)
        self.preview_button.toggled.connect(self.toggle_preview_view)
        view_row.addWidget(self.preview_button)
        mid.addLayout(view_row)
        self.grid_view = GridView()
        self.grid_view.editStarted.connect(self.push_history)
        self.grid_view.editFinished.connect(self.edited)
        self.grid_view.hovered.connect(self.hover_cell)
        self.grid_view.picked.connect(self.inspect_cell)
        self.grid_view.refresh()
        canvas_row = QHBoxLayout()
        self.region_legend = QLabel("<span style='color:#4876dc'>■</span> 定位图案<br><span style='color:#8a9bb4'>■</span> 分隔区域<br><span style='color:#1eaa91'>■</span> 时序图案<br><span style='color:#e8972c'>■</span> 格式信息<br><span style='color:#a55bcd'>■</span> 版本信息<br><span style='color:#e65782'>■</span> 校正图案<br><span style='color:#555555'>■</span> 数据与纠错码字")
        self.region_legend.setVisible(False)
        self.region_legend.setToolTip("区域提示颜色说明")
        canvas_row.addWidget(self.region_legend)
        canvas_row.addWidget(self.grid_view, 1)
        self.heat_legend = QWidget()
        legend_layout = QVBoxLayout(self.heat_legend)
        legend_layout.setContentsMargins(3,0,0,0)
        legend_layout.addWidget(QLabel("置信度"))
        legend_layout.addWidget(QLabel("100% 高"))
        self.legend_strip = QLabel()
        strip = np.transpose(confidence_legend(120,16)[:,::-1,:],(1,0,2)).copy()
        self.legend_strip.setPixmap(pixmap(strip))
        legend_layout.addWidget(self.legend_strip)
        legend_layout.addWidget(QLabel("0% 低"))
        legend_layout.addWidget(QLabel("区域：定位·时序·格式·版本·校正·数据"))
        legend_layout.addStretch()
        self.heat_legend.setToolTip("置信度 = |2p−1|，p 是图像黑色倾向；并非恢复内容的正确率。")
        self.heat_legend.hide()
        canvas_row.addWidget(self.heat_legend)
        mid.addLayout(canvas_row, 1)
        self.tool_bar = QToolBar("编辑工具")
        self.tool_bar.setMovable(False)
        self.tool_buttons = {}
        for title, glyph, modes in [("画笔", "✎", [("反转", "invert"), ("未知循环", "cycle")]),
                                     ("框选", "▣", [("框选", "select")]),
                                     ("点选", "⌖", [("块模式", "inspect_block"), ("格模式", "inspect_cell")])]:
            tool = QToolButton()
            icon_image = QPixmap(28, 28)
            icon_image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(icon_image)
            painter.setPen(QColor("#111111"))
            painter.setPen(QPen(QColor("#111111"), 2))
            if title == "画笔":
                painter.drawLine(6,22,21,7)
                painter.drawLine(9,24,24,9)
                painter.drawLine(6,22,9,24)
                painter.drawLine(21,7,24,9)
            elif title == "框选":
                painter.setPen(QPen(QColor("#111111"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(5,5,18,18)
            else:
                painter.drawEllipse(7,7,14,14)
                painter.drawLine(14,2,14,10)
                painter.drawLine(14,18,14,26)
                painter.drawLine(2,14,10,14)
                painter.drawLine(18,14,26,14)
            painter.end()
            tool.setIcon(QIcon(icon_image))
            tool.setIconSize(QSize(28,28))
            tool.setToolTip(title)
            tool.setCheckable(True)
            tool.setAccessibleName(title)
            menu = QMenu(tool)
            for label, mode in modes:
                action = menu.addAction(label)
                action.triggered.connect(lambda checked=False, m=mode, t=tool, name=title: self.activate_tool(m,t,name))
            tool.setMenu(menu)
            tool.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            tool.clicked.connect(lambda checked=False, t=tool, m=modes[0][1], name=title: self.activate_tool(getattr(t,"mode",m),t,name))
            self.tool_bar.addWidget(tool)
            self.tool_buttons[title] = tool
        mid.insertWidget(1, self.tool_bar)
        self.activate_tool("invert", self.tool_buttons["画笔"], "画笔")
        self.selection_actions = QWidget()
        row = QHBoxLayout(self.selection_actions)
        row.setContentsMargins(0,0,0,0)
        for label, value in [("设置为黑", 1), ("设置为白", 0), ("设置为未知", -1)]:
            row.addWidget(button(label, lambda checked=False, v=value: self.fill_selection(v)))
        self.selection_actions.hide()
        mid.insertWidget(2, self.selection_actions)
        row = QHBoxLayout()
        heat = QCheckBox("置信度热力图")
        self.heat_check = heat
        heat.setToolTip("置信度越高，红色越透明。")
        heat.toggled.connect(self.toggle_heat)
        row.addWidget(heat)
        lines = QCheckBox("网格线")
        lines.setChecked(True)
        lines.toggled.connect(lambda v: self.toggle_lines(v))
        row.addWidget(lines)
        regions = QCheckBox("区域提示")
        self.region_check = regions
        regions.setToolTip("标出定位、时序、格式、版本、校正和数据区域")
        regions.toggled.connect(self.toggle_regions)
        row.addWidget(regions)
        row.addWidget(button("适合窗口", self.grid_view.fit))
        row.setSpacing(3)
        controls.addLayout(row)
        row = QHBoxLayout()
        self.contour_check = QCheckBox("彩色等高线")
        self.contour_check.setToolTip("原图越黑越蓝，越白越白。")
        self.contour_check.toggled.connect(self.update_contours)
        row.addWidget(self.contour_check)
        self.contour_interval = QSpinBox()
        self.contour_interval.setRange(4, 128)
        self.contour_interval.setValue(16)
        self.contour_interval.setPrefix("间距 ")
        self.contour_interval.setToolTip("灰度级间距越小，等高线越精细。根据当前透视裁剪生成。")
        self.contour_interval.valueChanged.connect(self.update_contours)
        row.addWidget(self.contour_interval)
        self.contour_alpha = QSpinBox()
        self.contour_alpha.setRange(5, 90)
        self.contour_alpha.setValue(65)
        self.contour_alpha.setSuffix("%")
        self.contour_alpha.setToolTip("等高线不透明度")
        self.contour_alpha.valueChanged.connect(self.update_contours)
        row.addWidget(self.contour_alpha)
        row.setSpacing(3)
        controls.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button("补全固定结构", self.fix_structure))
        row.addWidget(button("下一可疑格", self.next_unknown))
        row.addWidget(button("解锁全部", self.unlock_all))
        row.setSpacing(3)
        controls.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button("旋转 90°", lambda: self.transform_grid("rotate")))
        row.addWidget(button("镜像", lambda: self.transform_grid("mirror")))
        row.addWidget(button("黑白反转", lambda: self.transform_grid("invert")))
        row.setSpacing(3)
        controls.addLayout(row)
        controls.addStretch()
        self.options_panel.setStyleSheet("QPushButton,QToolButton { padding: 5px 4px; } QSpinBox,QComboBox { padding: 3px; min-width: 0px; }")
        options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.cell_info = QLabel("")
        self.cell_info.setObjectName("muted")
        self.cell_info.setWordWrap(True)
        detail_row = QHBoxLayout()
        self.cell_patch = QLabel()
        self.contour_patch = QLabel()
        for title, patch in (("局部原图", self.cell_patch), ("局部等高线", self.contour_patch)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            patch.setFixedSize(108, 108)
            patch.setStyleSheet("background: #f8fbff; border: 1px solid #d4deeb; border-radius: 8px;")
            column.addWidget(patch)
            detail_row.addLayout(column)
        detail_row.addWidget(self.cell_info, 1)
        mid.addLayout(detail_row)
        self.inspection = QLabel()
        self.inspection.setWordWrap(True)
        self.inspection.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mid.addWidget(self.inspection)
        self.middle.setMinimumWidth(350)
        self.splitter.addWidget(self.middle)

        self.right = QWidget()
        self.right.setObjectName("rightPanel")
        right = QVBoxLayout(self.right)
        right.setContentsMargins(9, 0, 0, 0)
        label = QLabel("纠错")
        label.setObjectName("title")
        right.addWidget(label)
        self.tabs = QTabWidget()
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self.choose_sample)
        self.repair_list = QListWidget()
        self.repair_list.currentRowChanged.connect(self.preview_repair)
        self.tabs.addTab(self.sample_list, "网格候选")
        self.tabs.addTab(self.repair_list, "纠错候选")
        self.tabs.setMinimumHeight(145)
        self.tabs.setMaximumHeight(220)
        right.addWidget(self.tabs, 2)
        group = QGroupBox("纠错与搜索")
        group.setMinimumHeight(150)
        form = QFormLayout(group)
        self.deep = QCheckBox("深入搜索")
        self.deep.setChecked(True)
        form.addRow(self.deep)
        self.budget = QSpinBox()
        self.budget.setRange(2, 180)
        self.budget.setValue(15)
        self.budget.setSuffix(" 秒")
        form.addRow("搜索上限", self.budget)
        self.repair_button = button("分析纠错", self.start_repair, True)
        form.addRow(self.repair_button)
        right.addWidget(group)
        row = QHBoxLayout()
        self.apply_button = button("接受当前候选", self.accept_repair)
        self.apply_button.setEnabled(False)
        row.addWidget(self.apply_button)
        row.addWidget(button("返回编辑", self.cancel_preview))
        right.addLayout(row)
        self.result_title = QLabel("等待导入图片")
        self.result_title.setWordWrap(True)
        right.addWidget(self.result_title)
        self.content = QPlainTextEdit()
        self.content.setReadOnly(True)
        self.content.setPlaceholderText("解码内容")
        self.content.setMinimumHeight(95)
        right.addWidget(self.content, 1)
        row = QHBoxLayout()
        row.addWidget(button("复制内容", lambda: QApplication.clipboard().setText(self.content.toPlainText())))
        row.addWidget(button("导出数据", self.export_payload))
        row.addWidget(button("重新生成", self.regenerate))
        right.addLayout(row)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlaceholderText("校验结果")
        self.diagnostics.setMinimumHeight(95)
        self.diagnostics.hide()
        right.addWidget(self.diagnostics, 1)
        right.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setWidget(self.right)
        right_scroll.setMinimumWidth(320)
        self.splitter.addWidget(right_scroll)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([300, 780, 340])

        background_row = QHBoxLayout()
        self.colorful_background = QCheckBox("彩色作为背景")
        self.colorful_background.setChecked(True)
        background_row.addWidget(self.colorful_background)
        self.background_button = button("背景色", self.choose_background_color)
        background_row.addWidget(self.background_button)
        self.pick_button = button("图片取色", self.pick_background_color)
        background_row.addWidget(self.pick_button)
        background_row.addWidget(button("清除", lambda: self.set_background_color(None)))
        background_row.addStretch()
        root.addLayout(background_row)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        bottom.addWidget(self.progress, 1)
        self.cancel_button = button("停止搜索", self.cancel_job)
        self.cancel_button.setEnabled(False)
        bottom.addWidget(self.cancel_button)
        root.addLayout(bottom)
        self.setCentralWidget(central)
        for panel in (sampling, group, self.crop_view, self.grid_view):
            shadow = QGraphicsDropShadowEffect(panel)
            shadow.setBlurRadius(20)
            shadow.setOffset(4,5)
            shadow.setColor(QColor(134,151,177,65))
            panel.setGraphicsEffect(shadow)
        self.statusBar().showMessage("就绪")

    def adjust_selection(self):
        if self.image is None:
            return
        self.selection_dialog.show()
        self.source_view.enable_navigation(True)
        self.source_view.focus_selection()
        self.selection_dialog.raise_()

    def rotate_crop(self):
        if self.grid is not None and self.source_view.dimension:
            self.transform_grid("rotate")
        elif self.source_view.corners is not None:
            corners = np.roll(self.source_view.corners, -1, axis=0)
            self.source_view.set_image(self.image, corners)
            self.corners_changed(corners)

    def sync_source_preview(self):
        self.pages.setCurrentIndex(int(self.image is not None or self.grid is not None))
        p = self.source_view.corners
        self.crop_view.set_crop(self.image, p)
        self.crop_view.dimension = self.source_view.dimension
        self.corner_info.setText("" if p is None else
            "\n".join(f"{a} ({p[i,0]:.0f}, {p[i,1]:.0f})    {b} ({p[j,0]:.0f}, {p[j,1]:.0f})"
                      for a,i,b,j in (("左上",0,"右上",1),("左下",3,"右下",2))))

    def error(self, message):
        QMessageBox.warning(self, "无法完成", str(message))

    def choose_background_color(self):
        color = QColorDialog.getColor(QColor(*(self.background_color or (255,255,255))),self,"背景色")
        if color.isValid():
            self.set_background_color((color.red(),color.green(),color.blue()))

    def set_background_color(self, color):
        self.background_color = tuple(color) if color is not None else None
        self.source_view.picking_color = False
        self.source_view.unsetCursor()
        self.pick_button.setText("图片取色")
        if color is None:
            self.background_button.setText("背景色")
            self.background_button.setStyleSheet("")
        else:
            value = QColor(*color).name()
            foreground = "#fff" if sum(color)/3 < 130 else "#23344b"
            self.background_button.setText(value.upper())
            self.background_button.setStyleSheet(f"background:{value}; color:{foreground};")
        if self.image is not None:
            self.mark_dirty()

    def pick_background_color(self):
        if self.image is None:
            return
        self.adjust_selection()
        self.source_view.fit()
        self.source_view.picking_color = not self.source_view.picking_color
        self.pick_button.setText("取消取色" if self.source_view.picking_color else "图片取色")
        self.source_view.setCursor(Qt.CursorShape.CrossCursor if self.source_view.picking_color else Qt.CursorShape.ArrowCursor)

    def mark_dirty(self):
        self.dirty = True
        self.setWindowTitle(f"QR Repair Studio · {Path(self.project_path).name if self.project_path else '未保存工程'} *")

    def can_replace(self):
        if self.worker:
            self.error("请先停止并等待当前任务结束。")
            return False
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "保存当前工程", "当前工程尚未保存，是否先保存？",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return True

    def run_job(self, fn, done):
        if self.worker:
            return False
        self.check_timer.stop()
        self.pending_verify = False
        self.toolbar.setEnabled(False)
        for panel in (self.left, self.middle, self.right):
            panel.setEnabled(False)
        self.selection_dialog.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.worker = Worker(fn, self)
        w = self.worker
        w.progress.connect(self.statusBar().showMessage)
        w.result.connect(done)
        w.failed.connect(lambda text: self.error(text.splitlines()[-1]))
        w.finished.connect(self.job_finished)
        w.start()
        return True

    def job_finished(self):
        old = self.worker
        self.worker = None
        if old:
            old.deleteLater()
        self.toolbar.setEnabled(True)
        for panel in (self.left, self.middle, self.right):
            panel.setEnabled(True)
        self.selection_dialog.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.update_info()
        if self.pending_verify:
            self.pending_verify = False
            self.check_timer.start()

    def cancel_job(self):
        if self.worker:
            self.worker.stop.set()
            self.statusBar().showMessage("正在停止")
            self.cancel_button.setEnabled(False)

    def set_repair_diagnostics(self, lines, header=None):
        groups = {"通过": [], "未通过": [], "未检查": []}
        extras = []
        for line in lines or []:
            text = str(line)
            if text.startswith("块 ") and ": 通过" in text:
                groups["通过"].append(text)
            elif text.startswith("块 ") and ": 未通过" in text:
                groups["未通过"].append(text)
            elif text.startswith("块 ") and ": 未检查" in text:
                groups["未检查"].append(text)
            else:
                extras.append(text)
        out = []
        if header:
            out.append(header)
        for name in ("通过", "未通过", "未检查"):
            if groups[name]:
                if out: out.append("")
                out.append(name)
                out.extend(groups[name])
        if extras:
            if out: out.append("")
            out.extend(extras)
        self.diagnostics.setPlainText("\n".join(out))
        self.diagnostics.setVisible(bool(out))

    def clear_candidates(self):
        self.repairs = []
        self.selected_repair = None
        self.repair_list.clear()
        self.grid_view.preview = None
        self.apply_button.setEnabled(False)
        self.diagnostics.clear()
        self.diagnostics.hide()
        self.grid_view.repair_status = None
        self.grid_view.viewport().update()

    def reset_document(self):
        self.boundary_candidates = []
        self.region_list.blockSignals(True)
        self.region_list.clear()
        self.region_list.blockSignals(False)
        self.geometry_label.clear()
        self.source_view.finder_clicks = None
        for check in self.corner_locks:
            check.setChecked(False)
        self.selection_dialog.hide()
        self.preview_button.setChecked(False)
        self.check_timer.stop()
        self.history.clear()
        self.future.clear()
        self.clear_candidates()
        self.samples.clear()
        self.sample_list.clear()
        self.content.clear()
        self.diagnostics.clear()
        self.payload = None
        self.project_path = None
        self.grid = None
        self.inverted = False
        self.scan_geometry_mirrored = False
        self.grid_view.selection = None
        self.grid_view.contours = None
        self.contour_key = None
        self.contour_layer = None
        self.grid_view.set_grid(None)
        self.source_view.dimension = 0
        self.source_view.highlight = None
        self.last_suspect = None
        self.cell_patch.clear()
        self.contour_patch.clear()
        self.cell_info.clear()
        self.hovered_cell = None
        self.crop_view.highlight = None

    def import_image(self, path):
        if not self.can_replace():
            return
        try:
            image = read_image(path)
        except Exception as e:
            return self.error(e)
        self.reset_document()
        self.image = image
        self.reset_corners()
        self.mark_dirty()
        self.auto_locate()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入二维码图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*)")
        if path:
            self.import_image(path)

    def paste_image(self):
        q = QApplication.clipboard().image()
        if q.isNull():
            return self.error("剪贴板中没有图片。")
        if not self.can_replace():
            return
        from PySide6.QtGui import QImage
        q = q.convertToFormat(QImage.Format.Format_RGB888)
        image = np.frombuffer(q.bits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width()*3].reshape(q.height(), q.width(), 3).copy()
        self.reset_document()
        self.image = image
        self.reset_corners()
        self.mark_dirty()
        self.auto_locate()

    def reset_corners(self):
        if self.image is None:
            return
        self.scan_geometry_mirrored = False
        self.boundary_candidates = []
        self.region_list.blockSignals(True)
        self.region_list.clear()
        self.region_list.blockSignals(False)
        self.geometry_label.clear()
        for check in self.corner_locks:
            check.setChecked(False)
        h, w = self.image.shape[:2]
        p = np.array([[w*.1, h*.1], [w*.9, h*.1], [w*.9, h*.9], [w*.1, h*.9]], np.float32)
        self.source_view.set_image(self.image, p)
        self.source_view.dimension = 0
        self.source_view.highlight = None
        self.mark_dirty()
        self.invalidate_geometry()
        self.statusBar().showMessage("调整四角")

    def auto_locate(self, within_selection=False):
        if self.image is None:
            return self.error("请先导入图片。")
        image = self.image.copy()
        background = self.background_color
        colorful = self.colorful_background.isChecked()
        inverted = self.invert_input.isChecked()
        mirrored = self.mirror_input.isChecked()
        corners_in = self.source_view.corners.copy() if within_selection and self.source_view.corners is not None else None
        geometry_mirrored = self.scan_geometry_mirrored if corners_in is not None else False
        locked = np.array([check.isChecked() for check in self.corner_locks])

        def done(result):
            regions = result
            self.boundary_candidates = regions
            self.region_list.blockSignals(True)
            self.region_list.clear()
            for i, (corners, contents, sample, backend) in enumerate(regions):
                self.region_list.addItem(f"目标 {i+1} · {backend}" + (f" · {contents[0][0][:32]}" if contents else ""))
            self.region_list.blockSignals(False)
            if regions:
                corners, contents, sample, backend = regions[0]
            else:
                corners, contents, sample = None, [], None
            if corners is None:
                self.result_title.setText("保留当前选区" if within_selection else "请调整四角")
                self.statusBar().showMessage("定位未完成")
            else:
                self.source_view.set_image(self.image, corners)
                self.scan_geometry_mirrored = mirrored
                self.source_view.focus_selection()
                self.source_view.dimension = 0
                self.source_view.highlight = None
                self.mark_dirty()
                self.invalidate_geometry()
                self.result_title.setText("已定位")
                self.statusBar().showMessage("定位完成")
                if sample is not None and sample.geometry is not None:
                    self.show_geometry(sample)
                    self.orient_corner_locks(sample)
                if len(regions) > 1:
                    self.adjust_selection()
                    self.result_title.setText(f"找到 {len(regions)} 个目标，请选择二维码")
            if contents:
                self.content.setPlainText("\n\n".join(t for t, _ in contents))
                self.payload = contents[0][1]
                self.result_title.setText(f"识别完成")
        def detect_and_orient(cancel, progress):
            if corners_in is not None:
                corners, contents = optimize_quad(image, corners_in,background,colorful,inverted,cancel)
                # A manually supplied boundary can be refined even when no decoder
                # can detect the damaged symbol. Structural quality gates acceptance.
                proposals = [(corners if corners is not None else corners_in, contents, "选区精修")]
            else:
                proposals = [(d.corners, [(d.text,d.payload)] if d.decoded else [], d.backend)
                             for d in locate_candidates(image,background,colorful,inverted,cancel)]
            regions = []
            for corners, contents, backend in proposals:
                if cancel():
                    break
                corners = corners.copy()
                if corners_in is not None:
                    corners[locked] = corners_in[locked]
                progress("摆正选区")
                candidates = scan(image, corners, cancel=cancel, mirror=mirrored != geometry_mirrored,
                                   invert=inverted, locked_corners=locked if corners_in is not None else None)
                if candidates and candidates[0].score >= .68:
                    sample = candidates[0]
                    corners = oriented_corners(corners, sample)
                    regions.append((corners, contents, sample, backend))
            return regions
        self.run_job(detect_and_orient, done)

    def show_geometry(self, sample):
        quality = sample.geometry
        if quality is None:
            return
        self.geometry_status = (f"V{sample.grid.version} · {quality.message} · "
                                f"有效覆盖 {quality.coverage:.0%} · 结构吻合 {sample.score:.3f}")
        if quality.ambiguous:
            self.geometry_status += " · 版本候选接近，请检查网格"
        self.geometry_label.setText(self.geometry_status)
        self.corner_info.setToolTip(self.geometry_status)

    def orient_corner_locks(self, sample):
        locks = np.array([check.isChecked() for check in self.corner_locks])
        if sample.mirrored:
            locks = locks[[1,0,3,2]]
        locks = np.roll(locks, -sample.rotation)
        for check, value in zip(self.corner_locks, locks):
            check.setChecked(bool(value))

    def pick_finders(self):
        version = self.version.currentData()
        if not version:
            version, ok = QInputDialog.getInt(self, "三点定位", "二维码版本（用于确定定位图形与边界的距离）", 3, 1, 40)
            if not ok:
                return
        self.finder_version = version
        self.source_view.finder_clicks = []
        self.source_view.setCursor(Qt.CursorShape.CrossCursor)
        self.geometry_label.setText("依次点击左上、右上、左下定位图形的中心；完成后自动精修。")

    def finders_picked(self, centers):
        import cv2
        n = 17+4*self.finder_version
        ideal = np.float32([[3.5,3.5],[n-3.5,3.5],[3.5,n-3.5]])
        transform = cv2.getAffineTransform(ideal, centers)
        corners = cv2.transform(np.float32([[[0,0],[n,0],[n,n],[0,n]]]), transform)[0]
        from .engine import validate_quad
        try:
            validate_quad(corners)
        except ValueError:
            self.geometry_label.setText("三个中心点过于接近，请重新选择。")
            return
        self.version.setCurrentIndex(self.finder_version)
        self.source_view.set_image(self.image, corners)
        self.source_view.dimension = 0
        self.invalidate_geometry()
        self.mark_dirty()
        self.start_scan()

    def choose_boundary(self, index):
        if not 0 <= index < len(self.boundary_candidates):
            return
        corners, contents, sample, backend = self.boundary_candidates[index]
        self.source_view.set_image(self.image, corners)
        self.source_view.dimension = 0
        self.scan_geometry_mirrored = self.mirror_input.isChecked()
        self.invalidate_geometry()
        self.mark_dirty()
        self.source_view.focus_selection()
        self.show_geometry(sample)
        for check in self.corner_locks:
            check.setChecked(False)
        if contents:
            self.content.setPlainText(contents[0][0])
            self.payload = contents[0][1]

    def invalidate_geometry(self):
        self.check_timer.stop()
        self.samples.clear()
        self.sample_list.clear()
        self.clear_candidates()
        self.payload = None
        self.content.clear()
        self.diagnostics.clear()
        self.cell_patch.clear()
        self.contour_patch.clear()
        self.cell_info.clear()
        self.hovered_cell = None
        self.crop_view.highlight = None
        # Do not overlay a new crop on a grid sampled from old geometry.
        self.update_contours()

    def update_contours(self, *_):
        self.sync_source_preview()
        view = self.grid_view
        view.contour_visible = self.contour_check.isChecked()
        view.contour_opacity = self.contour_alpha.value()/100
        p = self.source_view.corners
        aligned = self.grid is None or self.source_view.dimension == len(self.grid.values)
        if self.image is None or p is None:
            self.contour_layer = None
            self.contour_key = None
        else:
            key = (id(self.image), p.tobytes(), self.contour_interval.value())
            if key != self.contour_key:
                try:
                    self.contour_layer = grayscale_contours(warp(self.image, p, 720), self.contour_interval.value())
                    self.contour_key = key
                except ValueError:
                    self.contour_layer = None
                    self.contour_key = None
        view.contours = self.contour_layer if aligned else None
        self.source_view.set_contours(self.contour_layer, view.contour_opacity, view.contour_visible)
        view.refresh()
        if getattr(self, "hovered_cell", None) is not None:
            self.hover_cell(*self.hovered_cell)

    def corners_changed(self, corners):
        self.mark_dirty()
        self.source_view.dimension = 0
        self.source_view.highlight = None
        self.invalidate_geometry()
        self.source_view.focus_selection()
        self.statusBar().showMessage("选区已更新")

    def start_scan(self):
        if self.image is None or self.source_view.corners is None:
            return self.error("请先导入图片并设置四角。")
        if self.grid is not None and np.any(np.isin(self.grid.source, [1,3])):
            answer = QMessageBox.question(self, "重新采样", "重新采样会替换当前网格。当前编辑可通过撤销恢复。继续？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        image, corners = self.image.copy(), self.source_view.corners.copy()
        if self.binary_mode:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            _, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = np.repeat(image[..., None], 3, axis=2)
        version, bias = self.version.currentData(), self.bias.value()
        # Mirrored geometry is already rectified after choosing a mirrored sample.
        mirror = self.mirror_input.isChecked() != self.scan_geometry_mirrored
        invert = self.invert_input.isChecked()
        locks = np.array([check.isChecked() for check in self.corner_locks])
        self.scan_corners = corners

        def done(samples):
            if not samples:
                self.statusBar().showMessage("采样已停止，没有生成候选。")
                return
            self.samples = samples[:8]
            self.sample_list.blockSignals(True)
            self.sample_list.clear()
            for s in self.samples:
                n = len(s.grid.values)
                transforms = (f" · 转 {s.rotation*90}°" if s.rotation else "") + (" · 镜像" if s.mirrored else "") + (" · 反色" if s.inverted else "")
                self.sample_list.addItem(f"V{s.grid.version}  /  {n}×{n}  ·  结构吻合 {s.score:.3f}{transforms}")
            self.sample_list.blockSignals(False)
            self.tabs.setCurrentIndex(0)
            self.sample_list.setCurrentRow(0)
            self.statusBar().showMessage("采样完成")
        self.run_job(lambda cancel, progress: scan(image, corners, [version] if version else None,
                     bias, 0, cancel, progress, mirror=mirror, invert=invert, locked_corners=locks), done)

    def choose_sample(self, index):
        if index < 0 or index >= len(self.samples):
            return
        self.push_history()
        sample = self.samples[index]
        self.grid = sample.grid.copy()
        self.inverted = sample.inverted
        corners = oriented_corners(self.scan_corners, sample)
        self.source_view.set_image(self.image, corners)
        self.scan_geometry_mirrored = self.mirror_input.isChecked()
        self.source_view.focus_selection()
        self.clear_candidates()
        self.grid_view.selection = None
        self.grid_view.set_grid(self.grid, True)
        self.source_view.dimension = len(self.grid.values)
        self.update_contours()
        self.show_geometry(sample)
        self.orient_corner_locks(sample)
        self.source_view.viewport().update()
        self.edited()

    def push_history(self):
        if self.grid is not None:
            corners = self.source_view.corners.copy() if self.source_view.corners is not None else None
            self.history.append((self.grid.copy(), corners, self.inverted, self.source_view.dimension, self.scan_geometry_mirrored))
            self.history = self.history[-80:]
            self.future.clear()

    def history_move(self, source, dest):
        if not source or self.worker:
            return
        p = self.source_view.corners.copy() if self.source_view.corners is not None else None
        dest.append((self.grid.copy(), p, self.inverted, self.source_view.dimension, self.scan_geometry_mirrored))
        self.grid, corners, self.inverted, dimension, self.scan_geometry_mirrored = source.pop()
        self.source_view.set_image(self.image, corners)
        self.source_view.dimension = dimension
        self.update_contours()
        self.grid_view.selection = None
        self.grid_view.set_grid(self.grid, True)
        self.edited()

    def undo(self):
        self.history_move(self.history, self.future)

    def redo(self):
        self.history_move(self.future, self.history)

    def update_info(self):
        self.sync_source_preview()
        self.undo_action.setEnabled(bool(self.history) and not self.worker)
        self.redo_action.setEnabled(bool(self.future) and not self.worker)
        self.apply_button.setEnabled(self.selected_repair is not None and not self.worker)
        if self.grid is None:
            self.summary.setText("")
        else:
            g = self.grid
            low = ((np.abs(g.probability-.5)*2 < .3) & (g.source == 0)).sum()
            self.summary.setText(f"V{g.version} · {len(g.values)}×{len(g.values)}  /  低置信 {int(low)}  /  锁定 {int(g.locked.sum())}")

    def edited(self):
        self.clear_candidates()
        self.payload = None
        self.content.clear()
        self.diagnostics.clear()
        self.result_title.setText("网格已修改，等待验证")
        self.grid_view.refresh()
        self.mark_dirty()
        self.update_info()
        if getattr(self, "hovered_cell", None) is not None:
            self.hover_cell(*self.hovered_cell)
        if self.worker:
            self.pending_verify = True
        else:
            self.check_timer.start()

    def quick_verify(self):
        if self.grid is None or self.worker:
            return
        if np.any(self.grid.values < 0):
            self.result_title.setText("仍有未知格 · 可运行擦除纠错或手动补全")
            return
        matrix = self.grid.values.copy()

        def done(result):
            self.diagnostics.setPlainText("\n".join(result.checks + result.errors))
            if result.valid:
                self.content.setPlainText(result.text)
                self.payload = result.payload
                self.result_title.setText("校验通过")
            else:
                self.payload = None
                self.content.clear()
                self.result_title.setText("待纠错")
        self.run_job(lambda cancel, progress: validate_matrix(matrix), done)

    def toggle_heat(self, value):
        self.grid_view.heatmap = value
        self.heat_legend.setVisible(value and not self.grid_view.preview_mode)
        self.grid_view.refresh()

    def toggle_preview_view(self, value):
        self.grid_view.preview_mode = value
        self.grid_view.dragging = False
        self.grid_view.rect_start = None
        self.preview_button.setText("返回编辑" if value else "预览视图")
        self.tool_bar.setVisible(not value)
        self.selection_actions.setVisible(not value and self.grid_view.tool == "select")
        self.heat_legend.setVisible(not value and self.heat_check.isChecked())
        self.grid_view.refresh()
        self.grid_view.fit()

    def toggle_binary(self):
        self.binary_mode = not self.binary_mode
        self.binary_button.setText("关闭二值化" if self.binary_mode else "二值化处理")
        if self.image is not None:
            self.sync_source_preview()
            self.start_scan()

    def activate_tool(self, mode, tool, title):
        tool.mode = mode
        tool.setToolTip(title + " · " + {"invert":"反转", "cycle":"未知 → 黑 → 白", "select":"框选", "inspect_block":"块模式", "inspect_cell":"格模式"}[mode])
        for item in self.tool_buttons.values():
            item.setChecked(item is tool)
        self.grid_view.tool = mode
        self.grid_view.inspected_mask = None
        if hasattr(self,"selection_actions"):
            self.selection_actions.setVisible(mode == "select")
        self.grid_view.viewport().update()

    def inspect_cell(self, r, c):
        from .inspection import inspect_module
        matrix = self.grid_view.preview if self.grid_view.preview is not None else self.grid.values
        mask, text = inspect_module(matrix, r, c, self.grid_view.tool == "inspect_block")
        self.grid_view.inspected_mask = mask
        self.inspection.setText(text)
        self.grid_view.viewport().update()

    def set_tool_mode(self, mode):
        if mode == "inspect":
            self.grid_view.tool = "inspect"
            self.inspect_mode = "single"
        elif mode == "unknown":
            self.grid_view.tool = "unknown"
        else:
            self.set_tool(mode)

    def set_tool(self, mode):
        self.grid_view.tool = mode
        self.selection_actions.setVisible(mode == "select" and not self.grid_view.preview_mode)
        if mode != "select":
            self.grid_view.selection = None
            self.grid_view.viewport().update()

    def toggle_lines(self, value):
        self.grid_view.show_lines = value
        self.grid_view.viewport().update()

    def toggle_regions(self, value):
        self.grid_view.show_regions = value
        self.region_legend.setVisible(value)
        self.grid_view.viewport().update()

    def ensure_editable(self):
        if self.grid is None:
            self.error("请先生成或载入网格。")
            return False
        self.cancel_preview()
        return True

    def fill_selection(self, value):
        if value not in (-1,0,1):
            return
        if not self.ensure_editable():
            return
        if not self.grid_view.selection:
            return self.error("请用「框选」工具拖动选中一片区域。")
        self.push_history()
        r0, c0, r1, c1 = self.grid_view.selection
        region = np.zeros_like(self.grid.locked)
        region[r0:r1+1, c0:c1+1] = True
        region &= ~self.grid.locked & ~finder_mask(self.grid.version)
        self.grid.values[region] = value
        self.grid.source[region] = 1
        self.edited()

    def fix_structure(self):
        if self.ensure_editable():
            self.push_history()
            self.grid = restore_structure(self.grid)
            self.grid_view.set_grid(self.grid)
            self.edited()

    def unlock_all(self):
        if self.ensure_editable():
            self.push_history()
            self.grid.locked[:] = False
            self.edited()

    def transform_grid(self, mode):
        if not self.ensure_editable():
            return
        self.push_history()
        if mode == "invert":
            self.grid.values = np.where(self.grid.values < 0, -1, 1-self.grid.values).astype(np.int8)
            self.grid.probability = 1-self.grid.probability
            self.inverted = not self.inverted
        else:
            fn = np.rot90 if mode == "rotate" else np.fliplr
            self.grid = Grid(*(fn(a).copy() for a in (self.grid.values, self.grid.probability, self.grid.locked, self.grid.source)))
            if self.source_view.corners is not None:
                p = self.source_view.corners
                p = np.roll(p, -1, axis=0) if mode == "rotate" else p[[1, 0, 3, 2]]
                if mode == "mirror":
                    self.scan_geometry_mirrored = not self.scan_geometry_mirrored
                self.source_view.set_image(self.image, p)
        self.grid_view.selection = None
        self.grid_view.set_grid(self.grid)
        self.source_view.dimension = len(self.grid.values)
        self.update_contours()
        self.edited()

    def hover_cell(self, r, c):
        if self.grid is None or not (0 <= r < len(self.grid.values) and 0 <= c < len(self.grid.values)):
            return
        self.hovered_cell = (r, c)
        g = self.grid
        names = ["原图采样", "人工修改", "固定结构", "纠错恢复"]
        value = {1: "黑", 0: "白", -1: "未知"}[int(g.values[r, c])]
        self.cell_info.setText(f"行 {r+1} · 列 {c+1}\n{value} · {names[min(int(g.source[r,c]),3)]}\n"
                              f"置信度 {abs(2*g.probability[r,c]-1):.0%}")
        import cv2
        n = len(g.values)
        self.contour_patch.clear()
        self.cell_patch.clear()
        self.crop_view.highlight = (r,c) if self.source_view.dimension else None
        self.crop_view.viewport().update()
        if self.source_view.dimension:
            self.source_view.highlight = (r, c)
            self.source_view.viewport().update()
            if self.image is not None and self.source_view.corners is not None:
                square = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
                h = cv2.getPerspectiveTransform(square, self.source_view.corners.astype(np.float32))
                local = np.array([[[c-1, r-1], [c+2, r-1], [c+2, r+2], [c-1, r+2]]], np.float32)
                pts = cv2.perspectiveTransform(local, h)[0]
                transform = cv2.getPerspectiveTransform(pts, np.array([[0,0],[108,0],[108,108],[0,108]], np.float32))
                patch = cv2.warpPerspective(self.image, transform, (108, 108), borderValue=(255,255,255))
                if self.contour_layer is not None:
                    # Sample the same filled bands shown in the editor, preserving
                    # quantized colors and alignment at the edge of the selection.
                    size = self.contour_layer.shape[0]
                    contour_local = np.array([[[c-2, r-2], [c+3, r-2], [c+3, r+3], [c-2, r+3]]], np.float32)
                    contour_local[..., 0] = np.clip(contour_local[..., 0], 0, n)
                    contour_local[..., 1] = np.clip(contour_local[..., 1], 0, n)
                    local_transform = cv2.getPerspectiveTransform(contour_local[0]*size/n,
                        np.float32([[0,0],[108,0],[108,108],[0,108]]))
                    contour = cv2.warpPerspective(self.contour_layer, local_transform, (108,108),
                        flags=cv2.INTER_NEAREST, borderValue=(12,45,105,255))
                    cv2.rectangle(contour,(36,36),(71,71),(230,151,39,255),1)
                    self.contour_patch.setPixmap(pixmap(contour))
                cv2.rectangle(patch, (36,36), (71,71), (230,151,39), 1)
                self.cell_patch.setPixmap(pixmap(patch))

    def next_unknown(self):
        if self.grid is None:
            return
        g = self.grid
        score = np.abs(g.probability-.5)*2
        score = score + (g.values >= 0)*.5 + g.locked*10 + fixed_mask(g.version)*2
        order = np.argsort(score.ravel())
        previous = getattr(self, "last_suspect", None)
        idx = 0
        if previous is not None:
            matches = np.flatnonzero(order == previous)
            if len(matches):
                idx = (int(matches[0])+1) % len(order)
        self.last_suspect = int(order[idx])
        r, c = divmod(self.last_suspect, len(g.values))
        self.grid_view.selection = (r, c, r, c)
        self.grid_view.centerOn((c+.5)*12, (r+.5)*12)
        self.grid_view.viewport().update()
        self.hover_cell(r, c)

    def start_repair(self):
        if self.grid is None:
            return self.error("请先生成网格。")
        self.cancel_preview()
        g, seconds, deep = self.grid.copy(), self.budget.value(), self.deep.isChecked()
        image = self.image.copy() if self.image is not None else None
        corners = self.source_view.corners.copy() if self.source_view.corners is not None else None
        if self.source_view.dimension != len(g.values):
            image = None
        inverted = self.inverted
        locks = np.array([check.isChecked() for check in self.corner_locks])

        def done(result):
            self.repairs, diagnostic, elapsed = result
            self.repair_list.blockSignals(True)
            self.repair_list.clear()
            for i, candidate in enumerate(self.repairs):
                self.repair_list.addItem(f"{i+1:02d}  {candidate.ec} / 掩码 {candidate.mask} · 吻合 {candidate.score:.3f}\n"
                    f"     改动 {candidate.changed} 格 · 高可靠度 {candidate.strong_changed} 格")
            self.repair_list.blockSignals(False)
            self.tabs.setCurrentIndex(1)
            self.set_repair_diagnostics(diagnostic)
            if self.repairs:
                self.repair_list.setCurrentRow(0)
            else:
                self.result_title.setText("暂无候选")
                self.content.clear()
                self.payload = None
            self.statusBar().showMessage(f"{elapsed:.1f} 秒 · {len(self.repairs)} 个候选")
        self.run_job(lambda cancel, progress: repair_with_geometry(g, image, corners, seconds, deep,
                     inverted, locks, cancel, progress), done)

    def preview_repair(self, index):
        if index < 0 or index >= len(self.repairs):
            return
        candidate = self.repairs[index]
        self.selected_repair = candidate
        self.grid_view.repair_status = "通过" if candidate.blocks and all(": 通过" in x for x in candidate.blocks if x.startswith("块 ")) else "未通过"
        self.grid_view.preview = candidate.matrix
        self.grid_view.refresh()
        self.content.setPlainText(candidate.text)
        self.payload = candidate.payload
        unique = len({r.payload for r in self.repairs})
        state = "存在不同内容的合法候选" if unique > 1 else "候选通过矩阵合法性校验"
        self.result_title.setText(f"{state} · 绿色为修改项")
        self.set_repair_diagnostics(candidate.blocks, f"图像吻合评分 {candidate.score:.3f}（非正确率）\n修改 {candidate.changed} 格，其中高可靠度 {candidate.strong_changed} 格")
        self.apply_button.setEnabled(not self.worker)

    def accept_repair(self):
        candidate = self.selected_repair
        if candidate is None:
            return
        self.push_history()
        if candidate.evidence is not None and candidate.corners is not None:
            self.grid = candidate.evidence.copy()
            self.source_view.set_image(self.image, candidate.corners)
            self.source_view.dimension = len(self.grid.values)
            self.update_contours()
        changed = self.grid.values != candidate.matrix
        self.grid.values = candidate.matrix.copy()
        self.grid.source[changed] = 3
        self.grid_view.set_grid(self.grid)
        self.edited()
        self.content.setPlainText(candidate.text)
        self.payload = candidate.payload
        self.set_repair_diagnostics(candidate.blocks)
        self.result_title.setText("已应用")

    def cancel_preview(self):
        had = self.selected_repair is not None
        self.selected_repair = None
        self.grid_view.preview = None
        self.grid_view.refresh()
        self.apply_button.setEnabled(False)
        self.diagnostics.clear()
        self.diagnostics.hide()
        if had:
            self.payload = None
            self.content.clear()
            self.result_title.setText("已返回原网格编辑")
            self.check_timer.start()

    def save(self, as_new=False):
        if self.image is None and self.grid is None:
            self.error("没有可保存的工程。")
            return False
        path = self.project_path
        if not path or as_new:
            path, _ = QFileDialog.getSaveFileName(self, "保存可编辑工程", path or "修复工程.qrr", "QR 修复工程 (*.qrr)")
        if not path:
            return False
        if not path.lower().endswith(".qrr"):
            path += ".qrr"
        try:
            save_project(path, self.image, self.source_view.corners, self.grid,
                {"bias": self.bias.value(),
                 "inverted": self.inverted, "dimension": self.source_view.dimension,
                 "mirror_input": self.mirror_input.isChecked(), "invert_input": self.invert_input.isChecked(),
                 "geometry_mirrored": self.scan_geometry_mirrored,
                 "corner_locks": [check.isChecked() for check in self.corner_locks],
                 "background_color": self.background_color, "colorful_background": self.colorful_background.isChecked(),
                 "contours": self.contour_check.isChecked(), "contour_interval": self.contour_interval.value(),
                 "contour_opacity": self.contour_alpha.value(), "overlay_style": 3,
                 "heatmap": self.heat_check.isChecked(), "regions": self.region_check.isChecked()})
        except Exception as e:
            self.error(e)
            return False
        self.project_path = path
        self.dirty = False
        self.setWindowTitle(f"QR Repair Studio · {Path(path).name}")
        self.statusBar().showMessage(f"已保存：{path}")
        return True

    def open_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "打开修复工程", "", "QR 修复工程 (*.qrr)")
        if not path or not self.can_replace():
            return
        try:
            image, corners, grid, settings = load_project(path)
            if grid is not None:
                grid = resolve_unknowns(grid)
        except Exception as e:
            return self.error(e)
        self.reset_document()
        self.image, self.grid = image, grid
        self.source_view.set_image(image, corners)
        self.source_view.dimension = settings.get("dimension", len(grid.values) if grid else 0)
        self.inverted = settings.get("inverted", False)
        self.mirror_input.setChecked(settings.get("mirror_input",False))
        self.invert_input.setChecked(settings.get("invert_input",False))
        self.scan_geometry_mirrored = settings.get("geometry_mirrored",False)
        for check, value in zip(self.corner_locks, settings.get("corner_locks", [False]*4)):
            check.setChecked(bool(value))
        self.set_background_color(settings.get("background_color"))
        self.colorful_background.setChecked(settings.get("colorful_background",True))
        self.bias.setValue(settings.get("bias", 0))
        self.contour_interval.setValue(settings.get("contour_interval",16))
        self.contour_alpha.setValue(settings.get("contour_opacity",65) if settings.get("overlay_style") == 3 else 65)
        self.contour_check.setChecked(settings.get("contours",False))
        self.heat_check.setChecked(settings.get("heatmap",False))
        self.region_check.setChecked(settings.get("regions",False))
        self.grid_view.set_grid(grid, True)
        self.update_contours()
        self.project_path = path
        self.dirty = False
        self.setWindowTitle(f"QR Repair Studio · {Path(path).name}")
        self.update_info()
        self.check_timer.start()

    def generate_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("生成二维码")
        dialog.resize(480, 340)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("输入文本或链接（UTF-8）"))
        text = QPlainTextEdit()
        layout.addWidget(text)
        ec = QComboBox()
        ec.addItems(["M", "L", "Q", "H"])
        layout.addWidget(ec)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                grid = generate(text.toPlainText().encode("utf-8"), ec.currentText())
            except Exception as e:
                return self.error(e)
            if not self.can_replace():
                return
            self.reset_document()
            self.image = None
            self.source_view.set_image(None)
            self.grid = grid
            self.grid_view.set_grid(grid, True)
            self.edited()

    def regenerate(self):
        if self.payload is None:
            return self.error("尚无有效解码数据。")
        payload = self.payload
        path, _ = QFileDialog.getSaveFileName(self, "按当前数据重新编码（图案可能变化）", "重新生成.png", "PNG 图片 (*.png);;SVG 矢量 (*.svg)")
        if not path:
            return
        try:
            matrix = generate(payload).values
            require_valid(matrix)
            if path.lower().endswith(".svg"):
                export_svg(path, matrix)
            else:
                if not path.lower().endswith(".png"):
                    path += ".png"
                Image.fromarray(render_matrix(matrix)).save(path)
            self.statusBar().showMessage("已按原始数据字节重新编码；采用 M 纠错级别，ECI 和原分段结构不保留。")
        except Exception as e:
            self.error(e)

    def export_dialog(self):
        if self.grid is None:
            return self.error("没有可导出的网格。")
        if self.selected_repair is not None:
            return self.error("请先接受纠错候选，或返回编辑后再导出当前网格。")
        path, selected = QFileDialog.getSaveFileName(self, "导出当前网格", "修复二维码.png", "PNG 图片 (*.png);;SVG 矢量 (*.svg);;模块矩阵 (*.json)")
        if not path:
            return
        a = self.grid.values
        try:
            if "json" in selected:
                if not path.lower().endswith(".json"):
                    path += ".json"
                Path(path).write_text(json.dumps({"version": self.grid.version, "modules": a.tolist(),
                                               "unknown": -1}, ensure_ascii=False), encoding="utf-8")
            else:
                require_valid(a)
                if "svg" in selected:
                    if not path.lower().endswith(".svg"):
                        path += ".svg"
                    export_svg(path, a)
                else:
                    if not path.lower().endswith(".png"):
                        path += ".png"
                    Image.fromarray(render_matrix(a)).save(path)
            self.statusBar().showMessage(f"已导出：{path}")
        except Exception as e:
            self.error(e)

    def export_payload(self):
        if self.payload is None:
            return self.error("尚无有效解码数据。")
        path, _ = QFileDialog.getSaveFileName(self, "导出解码原始字节", "二维码数据.bin", "二进制数据 (*.bin)")
        if path:
            try:
                Path(path).write_bytes(self.payload)
            except Exception as e:
                self.error(e)

    def help_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("使用说明与开源组件")
        dialog.resize(700, 610)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(f"""QR Repair Studio {__version__}

导入 → 调整四角 → 采样 → 纠错 → 导出

导入页可设置镜像和反色，默认关闭。选区自动摆正并适合窗口。
「调整选区」打开完整原图；滚轮缩放、中键移动，拖动四角调整。
底部可选背景色，也可从图片取色。「彩色作为背景」默认开启。
热图红色越淡，置信度越高。底部显示当前格的原图和等高线。
等高线默认不透明度 65%，开启后二维码淡化至 30%。
顶部「预览视图」显示带留白的黑白二维码。
选择「框选」后可填黑、填白；锁定格保留人工值。
Ctrl+Z 撤销，Ctrl+Y 重做，Ctrl+S 保存工程。

开源组件
PySide6 / Qt · OpenCV · ZXing-C++ · python-qrcode · reedsolo · NumPy · Pillow
许可证位于发行包 THIRD_PARTY 目录。
""")
        layout.addWidget(text)
        layout.addWidget(button("关闭", dialog.accept))
        dialog.exec()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.worker:
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self.open_project(path) if path.lower().endswith(".qrr") else self.import_image(path)

    def closeEvent(self, event):
        if self.worker:
            self.cancel_job()
            self.statusBar().showMessage("正在停止任务，请稍后再次关闭窗口。")
            event.ignore()
        elif self.can_replace():
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("QR Repair Studio")
    app.setOrganizationName("QR Repair Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.source_view.fit)
    QTimer.singleShot(0, window.grid_view.fit)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        QTimer.singleShot(150, lambda: window.open_project(path) if path.lower().endswith(".qrr") else window.import_image(path))
    sys.exit(app.exec())
