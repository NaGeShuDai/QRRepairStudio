"""Build a portable native Windows distribution using the active virtualenv."""
from pathlib import Path
import importlib.metadata as metadata
import shutil
import subprocess
import sys
import urllib.request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dist-dir", default="dist", help="Separate output directory when an older app is running")
options = parser.parse_args()

root = Path(__file__).parent
licenses = root / "THIRD_PARTY"
licenses.mkdir(exist_ok=True)
names = ["PySide6-Essentials", "shiboken6", "opencv-python-headless", "numpy", "Pillow", "zxing-cpp", "qrcode", "reedsolo"]
index = []
for name in names:
    dist = metadata.distribution(name)
    index.append(f"{name} {dist.version}\n{dist.metadata.get('Home-page','')}\nLicense: {dist.metadata.get('License-Expression') or dist.metadata.get('License','See bundled license')}\n")
    target = licenses / name
    target.mkdir(exist_ok=True)
    for file in dist.files or []:
        if any(word in str(file).lower() for word in ("license", "copying", "notice", "copyright")) and not str(file).endswith(".py"):
            src = Path(dist.locate_file(file))
            if src.is_file():
                dest = target / str(file).replace("/","_").replace("\\","_")
                shutil.copy2(src,dest)
(licenses / "INDEX.txt").write_text("\n".join(index),encoding="utf-8")
for name in ("LGPL-3.0-only.txt", "GPL-3.0-only.txt"):
    dest = licenses / "PySide6-Essentials" / name
    if not dest.exists():
        with urllib.request.urlopen("https://raw.githubusercontent.com/qt/qtbase/dev/LICENSES/"+name,timeout=30) as response:
            dest.write_bytes(response.read())
subprocess.run([sys.executable,"-m","PyInstaller","--noconfirm","--windowed","--onedir",
    "--distpath", str(root/options.dist_dir),
    "--name","QRRepairStudio","--collect-all","zxingcpp","--exclude-module","pytest",
    "--exclude-module","qrcode.tests",
    "--add-data", f"{licenses};THIRD_PARTY", str(root/"main.py")], cwd=root, check=True)
out = root / options.dist_dir / "QRRepairStudio"
# Qt 6.12 uses Windows' unversioned ICU API. PyInstaller may resolve a
# conflicting icuuc.dll from unrelated PATH entries (e.g. Poppler's ICU 78).
# Do not ship that foreign library over Windows' system implementation.
import pefile
foreign_icu = out / "_internal" / "icuuc.dll"
if foreign_icu.exists():
    pe = pefile.PE(str(foreign_icu), fast_load=True)
    pe.parse_data_directories(directories=[0])
    symbols = {s.name for s in pe.DIRECTORY_ENTRY_EXPORT.symbols}
    pe.close()
    if b"ucnv_open" not in symbols:
        foreign_icu.unlink()
        for dll in (out / "_internal").glob("icudt*.dll"):
            dll.unlink()
shutil.copytree(licenses,out/"THIRD_PARTY",dirs_exist_ok=True)
shutil.copytree(root/"samples",out/"samples",dirs_exist_ok=True)
source = out/"source"
source.mkdir(exist_ok=True)
shutil.copytree(root/"qrrepair",source/"qrrepair",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
for name in ("main.py", "requirements.txt", "requirements-dev.txt", "build.py", "make_samples.py", "README.md"):
    shutil.copy2(root/name,source/name)
(out/"使用说明.txt").write_text("双击 QRRepairStudio.exe 启动，无需安装 Python。\n\n导入图片 → 检查四角 → 生成网格候选 → 纠错 → 预览并接受 → 导出。\n程序内按 F1 查看完整帮助。samples 包含可复现的模糊透视图片和损坏工程。\n\n请保持整个文件夹完整，_internal 是运行依赖。工程保存为 .qrr，可跨次继续编辑。\n本程序使用 PySide6 / Qt 动态库；第三方许可证见 THIRD_PARTY。\n",encoding="utf-8")
print("Portable distribution:",out)
