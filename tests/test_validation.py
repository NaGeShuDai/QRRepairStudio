import numpy as np
import pytest
import qrcode
from reedsolo import RSCodec

from qrrepair.engine import (generate, Grid, repair, data_positions, template,
                             fixed_mask, render_matrix, decode_image, block_layout)
from qrrepair.validation import validate_matrix, validate_stream
from qrrepair.project import export_svg


@pytest.mark.parametrize("version", [1,2,6,7,9,10,26,27,40])
@pytest.mark.parametrize("ec", [0,1,2,3])
def test_all_version_groups_levels_and_reader_paths(version, ec):
    q = qrcode.QRCode(version=version, error_correction=ec, border=0, mask_pattern=version%8)
    q.add_data(b"12345")
    q.make(fit=False)
    result = validate_matrix(np.array(q.modules,np.int8))
    assert result.valid, result.errors
    assert result.payload == b"12345"


def test_readable_but_damaged_matrix_is_not_exportable(tmp_path):
    a = generate(b"Strict matrix validation", "H").values
    r,c = data_positions((len(a)-17)//4)[0]
    a[r,c] ^= 1
    assert decode_image(render_matrix(a), pure=True)
    result = validate_matrix(a)
    assert not result.valid and any("校验非零" in s for s in result.errors)
    with pytest.raises(ValueError):
        export_svg(tmp_path/"bad.svg",a)
    assert not (tmp_path/"bad.svg").exists()


@pytest.mark.parametrize("kind", ["finder","format","version","remainder"])
def test_exact_function_bits_and_remainder(kind):
    v = 7 if kind == "version" else 2
    q = qrcode.QRCode(version=v,error_correction=2,border=0)
    q.add_data(b"A")
    q.make(fit=False)
    a = np.array(q.modules,np.int8)
    if kind == "finder":
        r,c = 0,0
    elif kind == "version":
        r,c = 0,len(a)-11
    elif kind == "format":
        r,c = np.argwhere((template(v)>=0) & ~fixed_mask(v))[0]
    else:
        r,c = data_positions(v)[-1]
    a[r,c] ^= 1
    assert not validate_matrix(a).valid


def test_rs_valid_but_illegal_padding_is_rejected_by_inference():
    q = qrcode.QRCode(version=1,error_correction=1,border=0,mask_pattern=0)
    q.add_data(b"A")
    q.make(fit=False)
    block = block_layout(1,1)[0][0]
    data = bytearray(q.data_cache[:block.data_count])
    data[-1] ^= 1  # Valid segment, invalid final padding; recompute exact RS parity.
    q.data_cache = list(RSCodec(block.total_count-block.data_count).encode(data))
    q.makeImpl(False,0)
    a = np.array(q.modules,np.int8)
    result = validate_matrix(a)
    assert not result.valid and any("填充" in s for s in result.errors)
    g = Grid(a,np.where(a==1,.99,.01).astype(np.float32),np.ones_like(a,bool),np.zeros_like(a,np.uint8))
    candidates, _, _ = repair(g,seconds=3)
    assert not candidates


@pytest.mark.parametrize("data", [b"\xff", b"\x40\xff", b"\x00\x00", b"\x7f\xff\xff\xff"])
def test_malformed_segments(data):
    with pytest.raises(ValueError):
        validate_stream(data,1)
