from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from qrrepair.engine import generate, render_matrix, data_positions
from qrrepair.project import save_project

root = Path(__file__).parent / "samples"
root.mkdir(exist_ok=True)
payload = "QR Repair Studio | 本地二维码修复测试 | 2026-11-17"
g = generate(payload.encode("utf-8"), "H")
Image.fromarray(render_matrix(g.values, 12)).save(root / "01_原始二维码.png")
n = len(g.values)
clean = np.repeat(render_matrix(g.values, 12, 0)[:, :, None], 3, 2)
corners = np.float32([[95,75],[615,125],[555,650],[45,535]])
src = np.float32([[0,0],[n*12,0],[n*12,n*12],[0,n*12]])
h = cv2.getPerspectiveTransform(src, corners)
photo = cv2.warpPerspective(clean, h, (720,720), borderValue=(250,250,250))
photo = cv2.GaussianBlur(photo,(0,0),2.7)
gradient = np.linspace(.7,1,720)[None,:,None]
photo = np.clip(photo * gradient,0,255).astype(np.uint8)
Image.fromarray(photo).save(root / "02_模糊透视阴影.png")
damaged = g.copy()
for i in range(36):
    r,c = data_positions(g.version)[i*8]
    damaged.values[r,c] = -1 if i % 2 else 1-damaged.values[r,c]
    damaged.probability[r,c] = .49 if damaged.values[r,c] != 1 else .51
save_project(root / "03_损坏与未知格.qrr", photo, corners, damaged, {"dimension": n})
(root / "expected.json").write_text(json.dumps({"payload":payload,"version":g.version,
    "corners":corners.tolist(),"matrix":g.values.tolist()}, ensure_ascii=False, indent=2),encoding="utf-8")
print("Created reproducible samples, version",g.version)
