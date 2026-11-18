"""Compare a frozen pre-change engine with the full new detection/scan path.

Usage: python tools/benchmark_boundary.py --baseline .baseline/engine.py
Outputs independently generated cases and raw measurements; no timing assertions.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import qrcode
from qrrepair import engine


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("qrrepair._baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_case(index):
    version = [1, 3, 7, 12][index % 4]
    q = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_H, border=0,
                       mask_pattern=index % 8)
    q.add_data(f"CASE{index:03d}".encode())
    q.make(fit=False)
    original = np.array(q.modules, np.int8)
    matrix = original.copy()
    damage = index % 3 == 2
    rng = np.random.default_rng(20261119+index)
    if damage:
        coords = np.array(engine.data_positions(version))
        positions = coords[rng.choice(len(coords), len(coords)//4, replace=False)]
        matrix[positions[:,0], positions[:,1]] ^= 1
    n = len(matrix)
    scale = 8
    source = engine.render_matrix(matrix, scale, border=0)
    # Deterministic variation in perspective, size, illumination and blur.
    truth = np.float32([[65,45],[585,85],[550,600],[35,540]])
    truth += rng.uniform(-22,22,(4,2)).astype(np.float32)
    if index % 4 == 3:
        truth = (truth-truth.mean(0))*.65+truth.mean(0)
    src = np.float32([[0,0],[n*scale,0],[n*scale,n*scale],[0,n*scale]])
    image = cv2.warpPerspective(source, cv2.getPerspectiveTransform(src, truth), (660,660), borderValue=245)
    sigma = [0., .8, 1.6, 2.2][(index//4) % 4]
    if sigma:
        image = cv2.GaussianBlur(image,(0,0),sigma)
    illumination = np.linspace(.65,1,660)[None,:]
    image = np.clip(image*illumination+rng.normal(0,1.5,image.shape),0,255).astype(np.uint8)
    return np.repeat(image[:,:,None],3,2), truth, matrix, version, damage


def grid_error(corners, truth, n):
    axis = np.linspace(.5,n-.5,15,dtype=np.float32)
    x,y = np.meshgrid(axis,axis)
    pts = np.stack([x,y],-1).reshape(-1,1,2)
    square = np.float32([[0,0],[n,0],[n,n],[0,n]])
    actual = cv2.perspectiveTransform(pts,cv2.getPerspectiveTransform(square,corners))
    reference = cv2.perspectiveTransform(pts,cv2.getPerspectiveTransform(square,truth))
    pitch = np.mean(np.linalg.norm(np.roll(truth,-1,axis=0)-truth,axis=1))/n
    return float(np.linalg.norm(actual-reference,axis=2).mean()/pitch)


def measure(module, image, truth, matrix, version):
    start = time.perf_counter()
    corners, contents = module.locate(image)
    detected_at = time.perf_counter()
    result = {"detected":corners is not None, "detect_ms":(detected_at-start)*1000}
    if corners is not None:
        samples = module.scan(image,corners)
        best = samples[0]
        refined = best.corners if getattr(best,"corners",None) is not None else corners
        result.update(version=best.grid.version, grid_error=grid_error(refined,truth,len(matrix)),
                      bit_error=float(np.mean(best.grid.values != matrix)) if best.grid.version==version else 1.,
                      decoded=bool(module.decode_image(module.render_matrix(best.grid.values))))
    result["total_ms"] = (time.perf_counter()-start)*1000
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--count",type=int,default=32)
    parser.add_argument("--reuse-before", action="store_true", help="Reuse saved baseline measurements; rerun final code only")
    args = parser.parse_args()
    baseline = load_baseline(args.baseline)
    # Warm template caches equally; cold-start measured separately in report.
    warm = make_case(0)
    for module in (baseline,engine):
        measure(module,*warm[:4])
    rows = []
    previous = json.loads(Path('docs/boundary-benchmark.json').read_text(encoding='utf-8'))['cases'] if args.reuse_before else None
    for index in range(args.count):
        image,truth,matrix,version,damaged = make_case(index)
        row = {"case":index,"version":version,"damaged":damaged}
        for name,module in (("before",baseline),("after",engine)):
            row[name] = previous[index]['before'] if name == 'before' and previous else measure(module,image,truth,matrix,version)
        rows.append(row)
        print(json.dumps(row),flush=True)
    summary = {}
    for name in ("before","after"):
        records = [r[name] for r in rows]
        detected = [r for r in records if r["detected"]]
        summary[name] = {"detected":len(detected), "cases":len(rows),
            "correct_version":sum(r[name].get("version")==r["version"] for r in rows),
            "decoded_intact":sum(r[name].get("decoded",False) for r in rows if not r["damaged"]),
            "intact_cases":sum(not r["damaged"] for r in rows),
            "mean_grid_error_modules":float(np.mean([r['grid_error'] for r in detected])),
            "mean_bit_error":float(np.mean([r['bit_error'] for r in detected])),
            "median_total_ms":float(np.median([r['total_ms'] for r in records])),
            "p95_total_ms":float(np.percentile([r['total_ms'] for r in records],95))}
    output = Path('docs/boundary-benchmark.json')
    metadata = {"baseline_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        "seed":20261119,"count":args.count,"versions":[1,3,7,12],
        "dataset":"deterministic synthetic perspective, blur, shadow, noise; data damage in every third case",
        "measurement":"full-image locate plus scan; same inputs; warm template caches"}
    output.write_text(json.dumps({"metadata":metadata,"summary":summary,"cases":rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
