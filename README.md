# QR Repair Studio

本地二维码修复桌面工具。支持模糊、透视、旋转图像的网格采样、手动编辑与纠错。

1.6 边界升级：ZXing 校验失败候选与 Aruco 多码检测、结构掩码透视精修、版本筛选、原图直接采样与覆盖/位置稳定性降权。调整选区中可选择二维码、固定角点或点击三个定位图形中心。纠错失败时在剩余时间内尝试有原图结构证据支持的边界反馈。

实现与验收记录见 `docs/boundary-upgrade.md`；可复现基准在 `tools/benchmark_boundary.py`，逐例数据在 `docs/boundary-benchmark.json`。

![工作区](docs/workspace.png)

## 使用

从 [Releases](https://github.com/NaGeShuDai/QRRepairStudio/releases) 下载 Windows 压缩包，解压后运行 `QRRepairStudio.exe`。

导入图片 → 调整选区 → 生成网格 → 分析纠错 → 接受候选 → 导出。

- 自动定位三个定位码，摆正选区；支持镜像、反色和背景取色。
- 搜索版本 1–40，按黑白倾向生成网格，固定定位结构。
- 黑白绘制、框选填充、锁定、撤销与重做。
- 红色置信度热图、白至深蓝的填充等高线、局部原图对照。
- 黑白二维码预览，包含四格留白。
- Reed–Solomon 纠错，校验固定结构、格式、版本、数据段和填充位。
- 保存 `.qrr` 工程，导出 PNG、SVG、模块矩阵及解码数据。

滚轮缩放，中键平移。`Ctrl+O` 导入，`Ctrl+S` 保存，`Ctrl+Z` 撤销，`Ctrl+Y` 重做。

## 开发

Python 3.12，Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main.py
```

测试与打包：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe build.py
```

输出：`dist/QRRepairStudio/QRRepairStudio.exe`。`samples/` 包含原图、模糊透视图片和损坏工程。

采用 PySide6、OpenCV、ZXing-C++、python-qrcode、reedsolo、NumPy、Pillow。第三方许可证见 `THIRD_PARTY/`。
