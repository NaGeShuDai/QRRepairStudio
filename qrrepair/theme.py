STYLE = """
QMainWindow, QDialog, QWidget#rightPanel, QScrollArea, QScrollArea > QWidget > QWidget { background: #e9eef5; }
QWidget { color: #34435b; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 12px; }
QToolBar { background: #e9eef5; border: 0; padding: 10px; spacing: 8px; }
QToolButton, QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #f8fbff,stop:1 #dde5ef); border: 1px solid #d3dde9; border-top-color: white; border-left-color: white; border-radius: 9px; padding: 8px 11px; }
QToolButton:hover, QPushButton:hover { background: #f8fbff; color: #285ca9; }
QPushButton:pressed, QPushButton:checked { background: #dce7f7; border-color: #bfcfe3; color: #245b9c; }
QToolButton:disabled, QPushButton:disabled { color: #9ba8b9; background: #e6ebf2; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6e96db,stop:1 #4d76bf); color: white; border-color: #799dd8; font-weight: 600; }
QPushButton#primary:disabled { background: #c7d5eb; color: #f3f6fb; }
QLabel#title { font-size: 17px; font-weight: 600; color: #30425c; }
QLabel#muted { color: #7b899d; }
QGroupBox { background: #e9eef5; border: 1px solid #d5deea; border-top-color: white; border-radius: 12px; margin-top: 10px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #788aa4; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget { background: #f3f6fa; border: 1px solid #d4deeb; border-bottom-color: white; border-right-color: white; border-radius: 8px; padding: 6px; selection-background-color: #d6e5fa; selection-color: #285b9e; }
QComboBox QAbstractItemView { background: #f3f6fa; color: #34435b; selection-background-color: #d6e5fa; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #e2e8f0; }
QListWidget::item:selected { background: #dae6f6; color: #245b9c; border-radius: 6px; }
QSplitter::handle { background: #dce4ee; width: 2px; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: #e9eef5; padding: 8px 14px; color: #7b899d; }
QTabBar::tab:selected { background: #f8fbff; color: #285b9e; border-radius: 8px; }
QStatusBar { background: #e9eef5; color: #7b899d; }
QProgressBar { background: #dce4ee; border: 0; height: 4px; }
QProgressBar::chunk { background: #7198d9; }
QCheckBox::indicator { width: 15px; height: 15px; }
QToolTip { background: #f8fbff; color: #34435b; border: 1px solid #d4deeb; }
QScrollBar:vertical { background: #e9eef5; width: 10px; }
QScrollBar::handle:vertical { background: #c7d3e3; border-radius: 5px; min-height: 20px; }
"""
