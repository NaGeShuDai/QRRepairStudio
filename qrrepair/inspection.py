"""Read selected module evidence without changing or repairing it."""
import numpy as np
import cv2
from qrcode import util
from .engine import region_masks, template, fixed_mask, EC_NAMES, data_positions, block_layout


def inspect_module(matrix, r, c, block=True):
    a = np.asarray(matrix)
    version = (len(a)-17)//4
    masks = region_masks(version)
    name = next(k for k,v in masks.items() if v[r,c])
    labels = {"定位":"定位图案", "分隔":"分隔区域", "时序":"时序图案", "格式":"格式信息", "版本":"版本信息", "校正":"校正图案", "数据":"数据与纠错码字"}
    selected = np.zeros(a.shape,bool)
    if block:
        _, components = cv2.connectedComponents(masks[name].astype(np.uint8), connectivity=4)
        selected = components == components[r,c]
    else:
        selected[r,c] = True
    text = [f"{labels[name]} · 行 {r+1} / 列 {c+1}", "当前格：" + {-1:"未知",0:"白",1:"黑"}[int(a[r,c])]]
    fmt = (template(version)>=0) & ~fixed_mask(version)
    matches = [(ec,m) for ec in EC_NAMES for m in range(8) if np.array_equal(a[fmt],template(version,ec,m)[fmt])]
    if name == "格式":
        if len(matches)==1:
            ec,m = matches[0]
            rate = {"L":7,"M":15,"Q":25,"H":30}[EC_NAMES[ec]]
            text.append(f"纠错等级 {EC_NAMES[ec]} · 约 {rate}% 码字恢复能力 · 掩码 {m}")
        else:
            text.append("格式信息未完整读出")
    elif name == "数据":
        coords = list(data_positions(version))
        index = coords.index((r,c))
        word = index//8
        if len(matches)==1:
            ec,m = matches[0]
            blocks, mapping = block_layout(version,ec)
            owner = next(((bi,idx.index(word)) for bi,idx in enumerate(mapping) if word in idx), None)
            if owner is None:
                text.append("余数位")
            else:
                bi,local = owner
                spec = blocks[bi]
                role = "数据码字" if local < spec.data_count else "纠错码字"
                text.append(f"RS 块 {bi+1} · {role} {local+1} · 位 {index%8+1}")
                text.append(f"数据 {spec.data_count} 码字 / 纠错 {spec.total_count-spec.data_count} 码字")
                word_coords = coords[word*8:word*8+8]
                bits = [int(a[y,x]) for y,x in word_coords]
                if all(v>=0 for v in bits):
                    decoded = [v ^ int(util.mask_func(m)(y,x)) for v,(y,x) in zip(bits,word_coords)]
                    value = int("".join(map(str,decoded)),2)
                    text.append(f"原始码字 0x{value:02X} · 二进制 {value:08b}")
                else:
                    text.append("码字含未知位："+"".join("?" if v<0 else str(v) for v in bits))
                if block:
                    selected[:] = False
                    for wi in mapping[bi]:
                        for y,x in coords[wi*8:wi*8+8]: selected[y,x]=True
        else:
            text.append(f"交织码字 {word+1} · 位 {index%8+1} · 格式未读出，尚未确定 RS 分块")
            if block:
                selected[:] = False
                for y,x in coords[word*8:word*8+8]: selected[y,x]=True
    else:
        descriptions = {"定位":"用于定位二维码及判断方向", "分隔":"定位图案周围的白色分隔带", "时序":"黑白交替，用于确定网格间距", "版本":f"矩阵尺寸对应版本 V{version}", "校正":"用于修正透视与局部形变"}
        text.append(descriptions[name])
        expected = template(version)
        check = selected & (expected>=0)
        text.append(f"结构匹配 {int(np.sum(a[check]==expected[check]))}/{int(check.sum())} 格")
    return selected, "\n".join(text)
