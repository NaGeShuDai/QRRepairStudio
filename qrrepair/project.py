"""Versioned ZIP projects. No pickle, executable content or external image dependency."""
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image

from .engine import Grid, validate_quad


def save_project(path, image, corners, grid, metadata=None):
    path = Path(path)
    meta = {"schema": 1, "corners": np.asarray(corners).tolist() if corners is not None else None,
            "settings": metadata or {}, "has_image": image is not None, "has_grid": grid is not None}
    fd, tmp = tempfile.mkstemp(prefix=".qrrepair-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps(meta, ensure_ascii=False))
            if image is not None:
                buf = io.BytesIO()
                Image.fromarray(image).save(buf, "PNG")
                z.writestr("source.png", buf.getvalue())
            if grid is not None:
                buf = io.BytesIO()
                np.savez_compressed(buf, values=grid.values, probability=grid.probability,
                                    locked=grid.locked, source=grid.source)
                z.writestr("grid.npz", buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_project(path):
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > 300_000_000:
            raise ValueError("工程解压大小超过限制。")
        meta = json.loads(z.read("project.json"))
        if meta.get("schema") != 1:
            raise ValueError("不支持的工程版本。")
        image = None
        if meta.get("has_image"):
            with Image.open(io.BytesIO(z.read("source.png"))) as im:
                if im.width * im.height > 60_000_000:
                    raise ValueError("工程图片过大。")
                image = np.array(im.convert("RGB"))
        grid = None
        if meta.get("has_grid"):
            with np.load(io.BytesIO(z.read("grid.npz")), allow_pickle=False) as arrays:
                a = arrays["values"]
                if a.ndim != 2 or a.shape[0] != a.shape[1] or not 21 <= len(a) <= 177 or (len(a)-17) % 4:
                    raise ValueError("工程网格尺寸不合法。")
                if not np.isin(a, [-1, 0, 1]).all():
                    raise ValueError("工程网格值不合法。")
                p, lock, source = (arrays[k] for k in ("probability", "locked", "source"))
                if any(x.shape != a.shape for x in (p, lock, source)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError("工程数组不合法。")
                grid = Grid(a.astype(np.int8), p.astype(np.float32), lock.astype(bool), source.astype(np.uint8))
        corners = validate_quad(meta["corners"]) if meta.get("corners") is not None else None
        return image, corners, grid, meta.get("settings", {})


def export_svg(path, matrix, border=4):
    from .validation import require_valid
    require_valid(matrix)
    if border < 4:
        raise ValueError("二维码静区至少需要四个模块。")
    n = len(matrix) + border * 2
    cells = " ".join(f"M{c+border},{r+border}h1v1h-1z" for r, c in np.argwhere(matrix == 1))
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" shape-rendering="crispEdges"><rect width="{n}" height="{n}" fill="white"/><path d="{cells}" fill="black"/></svg>', encoding="utf-8")
