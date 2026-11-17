"""Strict Model 2 validation, without allowing a decoder to repair the input.

ZXing remains the payload interpreter; these checks independently enforce exact
function modules, format/version copies, syndromes, remainder bits and padding.
"""
from dataclasses import dataclass, field

import numpy as np
from qrcode import util
from reedsolo import RSCodec


@dataclass
class Validation:
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    text: str = ""
    payload: bytes = b""
    ec: int | None = None
    mask: int | None = None


def validate_stream(data, version):
    """Check segment boundaries, legal values and canonical termination/padding.

    Does not guess text encoding or rewrite payloads. Legacy nonstandard padding
    is intentionally rejected by the strict export gate, even if a reader accepts it.
    """
    bits = np.unpackbits(np.frombuffer(bytes(data), np.uint8))
    position = 0
    segments = 0
    group = 0 if version <= 9 else 1 if version <= 26 else 2
    has_sa, has_fnc, has_content = False, False, False

    def read(n):
        nonlocal position
        if position + n > len(bits):
            raise ValueError("数据段长度超出容量")
        value = 0
        for bit in bits[position:position+n]:
            value = (value << 1) | int(bit)
        position += n
        return value

    def zeroes(n):
        if read(n) != 0:
            raise ValueError("终止符或字节对齐位不是零")

    while position < len(bits):
        left = len(bits)-position
        if left < 4:
            zeroes(left)
            break
        mode = read(4)
        if mode == 0:
            zeroes((-position) % 8)
            i = 0
            while position < len(bits):
                if read(8) != (0xEC if i % 2 == 0 else 0x11):
                    raise ValueError("终止后的填充码字不符合 EC / 11 交替规则")
                i += 1
            break
        if mode == 3:  # Structured append precedes all other segments.
            if position != 4 or has_sa:
                raise ValueError("结构化追加头的位置不合法")
            index, total = read(4), read(4)+1
            read(8)
            if total < 2 or index >= total:
                raise ValueError("结构化追加编号不合法")
            has_sa = True
            continue
        if mode in (5, 9):
            if has_fnc or has_content:
                raise ValueError("FNC1 头重复或位置不合法")
            has_fnc = True
            if mode == 9:
                read(8)
            continue
        if mode == 7:
            first = read(8)
            if first & 0x80 == 0:
                value = first
            elif first & 0xC0 == 0x80:
                value = ((first & 0x3F) << 8) | read(8)
                if value < 128:
                    raise ValueError("ECI 长度编码不合法")
            elif first & 0xE0 == 0xC0:
                value = ((first & 0x1F) << 16) | read(16)
                if value < 16384:
                    raise ValueError("ECI 长度编码不合法")
            else:
                raise ValueError("ECI 前缀不合法")
            if value > 999999:
                raise ValueError("ECI 编号超出范围")
            continue
        widths = {1:(10,12,14), 2:(9,11,13), 4:(8,16,16), 8:(8,10,12), 13:(8,10,12)}
        if mode not in widths:
            raise ValueError(f"不合法的数据模式 {mode}")
        if mode == 13 and read(4) != 1:
            raise ValueError("不支持的汉字子集")
        count = read(widths[mode][group])
        segments += 1
        has_content = True
        if mode == 1:
            while count:
                n = min(count, 3)
                if read({1:4, 2:7, 3:10}[n]) >= 10**n:
                    raise ValueError("数字段值超出范围")
                count -= n
        elif mode == 2:
            while count >= 2:
                if read(11) >= 45*45:
                    raise ValueError("字母数字段值超出范围")
                count -= 2
            if count and read(6) >= 45:
                raise ValueError("字母数字段值超出范围")
        elif mode == 4:
            if position + count*8 > len(bits):
                raise ValueError("字节段长度超出容量")
            position += count*8
        else:
            for _ in range(count):
                value = read(13)
                if mode == 8:
                    code = ((value//0xC0)<<8) | (value % 0xC0)
                    code += 0x8140 if code < 0x1F00 else 0xC140
                    lead, trail = code>>8, code & 255
                    if not ((0x81 <= lead <= 0x9F or 0xE0 <= lead <= 0xEB) and
                            (0x40 <= trail <= 0xFC and trail != 0x7F) and code <= 0xEBBF):
                        raise ValueError("Kanji 段字符范围不合法")
                else:
                    code = ((value//0x60)<<8) | (value % 0x60)
                    code += 0xA1A1 if code < 0x0A00 else 0xA6A1
                    lead, trail = code>>8, code & 255
                    if not ((0xA1 <= lead <= 0xAA or 0xB0 <= lead <= 0xFA) and 0xA1 <= trail <= 0xFE):
                        raise ValueError("汉字段字符范围不合法")
    return segments


def validate_matrix(matrix, reader_check=True):
    from .engine import template, fixed_mask, data_positions, block_layout, EC_NAMES, decode_image, render_matrix
    result = Validation()
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 21 <= len(a) <= 177 or (len(a)-17) % 4:
        result.errors.append("矩阵尺寸不符合标准 QR V1–40")
        return result
    if not np.isin(a, [0,1]).all():
        result.errors.append("存在未知格或非黑白模块")
        return result
    v = (len(a)-17)//4
    fixed = fixed_mask(v)
    mismatch = int(np.count_nonzero(a[fixed] != template(v)[fixed]))
    if mismatch:
        result.errors.append(f"定位、分隔、定时、校正或版本信息有 {mismatch} 格不符")
    else:
        result.checks.append("固定结构与版本信息完全一致")
    fmt = (template(v) >= 0) & ~fixed
    matches = [(ec,m) for ec in EC_NAMES for m in range(8) if np.array_equal(a[fmt],template(v,ec,m)[fmt])]
    if len(matches) != 1:
        result.errors.append("两份格式信息不一致或 BCH 格式码非法")
        return result
    ec, mask = matches[0]
    result.ec, result.mask = ec, mask
    result.checks.append(f"两份格式码精确一致 · {EC_NAMES[ec]} / 掩码 {mask}")
    coords = np.array(data_positions(v))
    masks = np.array([util.mask_func(mask)(int(r),int(c)) for r,c in coords],np.uint8)
    bits = a[coords[:,0],coords[:,1]].astype(np.uint8) ^ masks
    blocks, mapping = block_layout(v,ec)
    count = sum(b.total_count for b in blocks)
    if np.any(bits[count*8:]):
        result.errors.append("解除掩码后的余数位不是零")
    else:
        result.checks.append("余数位为零")
    raw = np.packbits(bits[:count*8])
    data = bytearray()
    for i, (b,indices) in enumerate(zip(blocks,mapping)):
        code = raw[indices]
        if not all(RSCodec(b.total_count-b.data_count).check(bytearray(code))):
            result.errors.append(f"块 {i+1} 校验非零（仍依赖阅读器纠错）")
        data.extend(code[:b.data_count].tolist())
    if result.errors:
        return result
    result.checks.append(f"全部 {len(blocks)} 个 RS 块校验为零，无需再纠错")
    try:
        segments = validate_stream(data,v)
        result.checks.append(f"{segments} 个数据段、长度、终止符及填充合法")
    except ValueError as e:
        result.errors.append(str(e))
        return result
    if reader_check:
        # A pure reader can accept a matrix that a normal camera detector rejects.
        pure = decode_image(render_matrix(a,6),pure=True)
        normal = decode_image(render_matrix(a,8),pure=False)
        if not pure or not normal or not pure[0].valid or not normal[0].valid:
            result.errors.append("带四模块静区的普通识别或矩阵识别未通过")
            return result
        if bytes(pure[0].bytes) != bytes(normal[0].bytes):
            result.errors.append("两种识别路径的数据不一致")
            return result
        result.text, result.payload = pure[0].text, bytes(pure[0].bytes)
        result.checks.append("普通检测与矩阵检测结果一致（含四模块静区）")
    result.valid = True
    return result


def require_valid(matrix):
    result = validate_matrix(matrix)
    if not result.valid:
        raise ValueError("不能导出为合法二维码：\n"+"\n".join(result.errors)+"\n请先纠错；未完成的网格可保存工程或导出 JSON。")
    return result
