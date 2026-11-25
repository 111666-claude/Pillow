from __future__ import annotations

import struct
from collections.abc import Callable
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pytest

from PIL import Image

from .helper import assert_image_equal_tofile

if TYPE_CHECKING:
    from pathlib import Path

    import olefile

FpxImagePlugin = pytest.importorskip(
    "PIL.FpxImagePlugin", reason="olefile not installed"
)

TEST_FILE = "Tests/images/input_bw_one_band.fpx"
HEADER_STREAM = [
    "Data Object Store 000001",
    "Resolution 0001",
    "Subimage 0000 Header",
]
CONTENTS_STREAM = ["Data Object Store 000001", "\x05Image Contents"]
# Offset of the highest-resolution width property (0x1000002) inside the
# Image Contents property set stream.
_WIDTH_PROPERTY_OFFSET = 212

Mutator = Callable[["olefile.OleFileIO[Any]"], None]


def _mutated_fpx(tmp_path: Path, mutate: Mutator) -> BytesIO:
    import olefile

    path = tmp_path / "mutated.fpx"
    with open(TEST_FILE, "rb") as f:
        path.write_bytes(f.read())

    ole = olefile.OleFileIO(str(path), write_mode=True)
    try:
        mutate(ole)
    finally:
        ole.close()
    return BytesIO(path.read_bytes())


def _set_header_field(
    ole: olefile.OleFileIO[Any], field_offset: int, value: int
) -> None:
    # field_offset is relative to the 36-byte header after the 28-byte prefix
    data = bytearray(ole.openstream(HEADER_STREAM).read())
    struct.pack_into("<i", data, 28 + field_offset, value)
    ole.write_stream(HEADER_STREAM, bytes(data))


def _set_width_property(ole: olefile.OleFileIO[Any], value: int) -> None:
    data = bytearray(ole.openstream(CONTENTS_STREAM).read())
    struct.pack_into("<I", data, _WIDTH_PROPERTY_OFFSET, value)
    ole.write_stream(CONTENTS_STREAM, bytes(data))


def _set_vector_property_count(
    ole: olefile.OleFileIO[Any], property_id: int, count: int
) -> None:
    # Inflate the element count of a VT_VECTOR property. olefile's parser
    # loops over every element, so an unchecked count can hang the decoder.
    data = bytearray(ole.openstream(CONTENTS_STREAM).read())
    section_offset = struct.unpack_from("<I", data, 44)[0]
    property_count = struct.unpack_from("<I", data, section_offset + 4)[0]
    for i in range(property_count):
        entry = section_offset + 8 + i * 8
        if struct.unpack_from("<I", data, entry)[0] != property_id:
            continue
        value_offset = struct.unpack_from("<I", data, entry + 4)[0]
        count_offset = section_offset + value_offset + 4
        struct.pack_into("<I", data, count_offset, count)
        ole.write_stream(CONTENTS_STREAM, bytes(data))
        return
    msg = f"property {property_id:#x} not found"
    raise AssertionError(msg)


def test_sanity() -> None:
    with Image.open(TEST_FILE) as im:
        assert im.mode == "L"
        assert im.size == (70, 46)
        assert im.format == "FPX"

        assert_image_equal_tofile(im, "Tests/images/input_bw_one_band.png")


def test_close() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, FpxImagePlugin.FpxImageFile)
    assert im.ole.fp.closed

    im = Image.open(TEST_FILE)
    assert isinstance(im, FpxImagePlugin.FpxImageFile)
    im.close()
    assert im.ole.fp.closed


def test_invalid_file() -> None:
    # Test an invalid OLE file
    invalid_file = "Tests/images/flower.jpg"
    with pytest.raises(SyntaxError):
        FpxImagePlugin.FpxImageFile(invalid_file)

    # Test a valid OLE file, but not an FPX file
    ole_file = "Tests/images/test-ole-file.doc"
    with pytest.raises(SyntaxError):
        FpxImagePlugin.FpxImageFile(ole_file)


def test_invalid_tile_size() -> None:
    with open(TEST_FILE, "rb") as f:
        data = f.read()
    data = data[:4204] + b"\x00" * 8 + data[4212:]
    with pytest.raises(ValueError, match="Tile must be 64 pixels by 64 pixels"):
        Image.open(BytesIO(data))


def test_invalid_number_of_bands() -> None:
    with pytest.raises(OSError, match="Invalid number of bands"):
        with Image.open("Tests/images/input_bw_five_bands.fpx"):
            pass


def test_oob_write() -> None:
    with pytest.raises(ValueError, match="Tile must be 64 pixels by 64 pixels"):
        Image.open("Tests/images/GHSA-5x8w-8jj5-q52h_oob_fpx_write.fpx")

    for img_path in (
        "Tests/images/GHSA-5x8w-8jj5-q52h_oob64.fpx",
        "Tests/images/GHSA-5x8w-8jj5-q52h_wide.fpx",
    ):
        with Image.open(img_path) as im:
            with pytest.raises(OSError, match="broken data stream"):
                im.load()


@pytest.mark.parametrize(
    "value, match",
    [
        # A negative width is stored as an unsigned 32-bit value by olefile
        (0xFFFFFFFF, "image dimensions must be positive"),
        (0, "image dimensions must be positive"),
    ],
)
def test_invalid_dimensions(tmp_path: Path, value: int, match: str) -> None:
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_width_property(ole, value)

    with pytest.raises(OSError, match=match):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_huge_dimensions(tmp_path: Path) -> None:
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_width_property(ole, 0x40000000)

    with pytest.raises(Image.DecompressionBombError):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_inflated_tile_count(tmp_path: Path) -> None:
    # tilecount is header field 3; an inflated count used to make the parser
    # read far past the header stream
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_header_field(ole, 12, 0x40000000)

    with pytest.raises(OSError, match="tile directory extends past"):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_zero_tile_count(tmp_path: Path) -> None:
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_header_field(ole, 12, 0)

    with pytest.raises(OSError, match="invalid tile count"):
        Image.open(_mutated_fpx(tmp_path, mutate))


@pytest.mark.parametrize("length", [0, 8])
def test_short_tile_entries(tmp_path: Path, length: int) -> None:
    # entry length is header field 8 (32 bytes after the prefix); entries are
    # indexed at offsets up to 15, so anything below 16 is invalid
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_header_field(ole, 32, length)

    with pytest.raises(OSError, match="invalid tile entry length"):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_tile_directory_offset(tmp_path: Path) -> None:
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        # directory offset is header field 7
        _set_header_field(ole, 28, -4)

    with pytest.raises(OSError, match="tile directory"):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_inflated_vector_property_count(tmp_path: Path) -> None:
    # An inflated VT_VECTOR element count used to make olefile's property
    # parser loop for a very long time; it must be rejected before parsing.
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        _set_vector_property_count(ole, 0x2010003, 0x7FFFFFFF)

    with pytest.raises(OSError, match="vector property extends past"):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_invalid_property_section_count(tmp_path: Path) -> None:
    def mutate(ole: olefile.OleFileIO[Any]) -> None:
        # The section count lives at offset 24 of the property set header
        data = bytearray(ole.openstream(CONTENTS_STREAM).read())
        struct.pack_into("<I", data, 24, 0x7FFFFFFF)
        ole.write_stream(CONTENTS_STREAM, bytes(data))

    with pytest.raises(OSError, match="invalid property set section count"):
        Image.open(_mutated_fpx(tmp_path, mutate))


def test_tile_count_grid_mismatch() -> None:
    # A directory that fits inside the stream but whose tile count does not
    # match the grid implied by the image dimensions must be rejected rather
    # than parsed until the grid walk ends.
    prefix = b"\xfe\xff" + b"\x00" * 26
    fields = struct.pack("<9i", 36, 70, 46, 3, 64, 64, 1, 36, 16)
    entries = b"".join(
        struct.pack("<i", 100 + i * 64) + struct.pack("<i", 0) + b"\x00" * 8
        for i in range(3)
    )
    header = prefix + fields + entries

    class FakeOLE:
        def get_size(self, stream_name: object) -> int:
            return len(header)

        def openstream(self, stream_name: object) -> BytesIO:
            return BytesIO(header)

    im = FpxImagePlugin.FpxImageFile.__new__(FpxImagePlugin.FpxImageFile)
    im._size = (70, 46)
    im.rawmode = "L"
    im.jpeg = {}
    im.ole = FakeOLE()

    with pytest.raises(OSError, match="tile count does not match the image grid"):
        im._open_subimage()


def test_valid_multiresolution_edge_tiles() -> None:
    # 70x46 needs a 2x1 tile grid at full resolution (the second tile only
    # covers 6 pixels) and a 1x1 grid (35x23) at the lower resolution; both
    # subimages must decode with their original pixels and metadata
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, FpxImagePlugin.FpxImageFile)
        assert im.size == (70, 46)
        assert im.mode == "L"
        assert im.maxid == 1
        assert len(im.tile) == 2
        assert im.tile[0].extents == (0, 0, 64, 46)
        assert im.tile[1].extents == (64, 0, 70, 46)

        assert_image_equal_tofile(im, "Tests/images/input_bw_one_band.png")
