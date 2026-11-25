from __future__ import annotations

import struct
from io import BytesIO

import pytest

from PIL import BlpImagePlugin, Image

from .helper import (
    assert_image_equal_tofile,
    assert_image_similar_tofile,
    hopper,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_load_blp1() -> None:
    with Image.open("Tests/images/blp/blp1_jpeg.blp") as im:
        assert_image_equal_tofile(im, "Tests/images/blp/blp1_jpeg.png")

    with Image.open("Tests/images/blp/blp1_jpeg2.blp") as im:
        assert im.mode == "RGBA"
        im.load()


def test_load_blp2_raw() -> None:
    with Image.open("Tests/images/blp/blp2_raw.blp") as im:
        assert_image_equal_tofile(im, "Tests/images/blp/blp2_raw.png")


def test_load_blp2_dxt1() -> None:
    with Image.open("Tests/images/blp/blp2_dxt1.blp") as im:
        assert_image_equal_tofile(im, "Tests/images/blp/blp2_dxt1.png")


def test_load_blp2_dxt1a() -> None:
    with Image.open("Tests/images/blp/blp2_dxt1a.blp") as im:
        assert_image_equal_tofile(im, "Tests/images/blp/blp2_dxt1a.png")


def _blp2_dxt(
    width: int,
    height: int,
    alpha_encoding: BlpImagePlugin.AlphaEncoding,
    alpha: bool,
    blocks: bytes,
) -> BytesIO:
    header = (
        b"BLP2"
        + struct.pack("<i", 1)  # uncompressed or DXT
        + struct.pack(
            "<bbbb", BlpImagePlugin.Encoding.DXT, int(alpha), alpha_encoding, 0
        )
        + struct.pack("<II", width, height)
    )
    palette = bytes(256 * 4)
    offset = len(header) + 16 * 4 * 2 + len(palette)
    header += struct.pack("<16I", offset, *([0] * 15))
    header += struct.pack("<16I", len(blocks), *([0] * 15))
    return BytesIO(header + palette + blocks)


def _dxt1_color(color0: int, color1: int, control: int) -> tuple[int, int, int]:
    r0, g0, b0 = BlpImagePlugin.unpack_565(color0)
    r1, g1, b1 = BlpImagePlugin.unpack_565(color1)
    if control == 0:
        return r0, g0, b0
    if control == 1:
        return r1, g1, b1
    if color0 > color1:
        if control == 2:
            return (2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3
        return (2 * r1 + r0) // 3, (2 * g1 + g0) // 3, (2 * b1 + b0) // 3
    if control == 2:
        return (r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2
    return 0, 0, 0


# A different control op for each pixel of a block, so that padding pixels
# never decode to the same colour as the pixels that are inside the image.
CONTROLS = [(i + j) % 4 for j in range(4) for i in range(4)]


def _dxt1_block(color0: int, color1: int) -> bytes:
    bits = sum(control << (2 * index) for index, control in enumerate(CONTROLS))
    return struct.pack("<HHI", color0, color1, bits)


@pytest.mark.parametrize("size", [(1, 2), (2, 4), (3, 5), (5, 4), (4, 4), (3, 8)])
def test_load_blp2_dxt1_narrow(size: tuple[int, int]) -> None:
    width, height = size
    blocks_per_row = (width + 3) // 4
    blocks_per_column = (height + 3) // 4
    colors = [(0xF800, 0x001F), (0x07E0, 0x001F)]
    blocks = b"".join(
        _dxt1_block(*colors[by % len(colors)])
        for by in range(blocks_per_column)
        for _ in range(blocks_per_row)
    )

    with Image.open(
        _blp2_dxt(width, height, BlpImagePlugin.AlphaEncoding.DXT1, False, blocks)
    ) as im:
        im.load()
        assert im.mode == "RGB"
        assert len(im.tobytes()) == width * height * 3
        for y in range(height):
            color0, color1 = colors[(y // 4) % len(colors)]
            for x in range(width):
                control = CONTROLS[(y % 4) * 4 + (x % 4)]
                assert im.getpixel((x, y)) == _dxt1_color(color0, color1, control)


def test_load_blp2_dxt1_narrow_alpha() -> None:
    width, height = 2, 4
    blocks = _dxt1_block(0x001F, 0xF800)

    with Image.open(
        _blp2_dxt(width, height, BlpImagePlugin.AlphaEncoding.DXT1, True, blocks)
    ) as im:
        im.load()
        assert im.mode == "RGBA"
        assert len(im.tobytes()) == width * height * 4
        for y in range(height):
            for x in range(width):
                control = CONTROLS[(y % 4) * 4 + (x % 4)]
                expected = _dxt1_color(0x001F, 0xF800, control)
                # With color0 <= color1, control 3 is transparent black
                alpha = 0 if control == 3 else 255
                assert im.getpixel((x, y)) == (*expected, alpha)


def _dxt3_block(alpha_nibbles: list[int], color0: int, color1: int) -> bytes:
    alpha = bytes(
        alpha_nibbles[i] | (alpha_nibbles[i + 1] << 4) for i in range(0, 16, 2)
    )
    bits = sum(control << (2 * index) for index, control in enumerate(CONTROLS))
    return alpha + struct.pack("<HHI", color0, color1, bits)


def _dxt5_block(
    codes: list[int], alpha0: int, alpha1: int, color0: int, color1: int
) -> bytes:
    alpha = sum(code << (3 * index) for index, code in enumerate(codes))
    bits = sum(control << (2 * index) for index, control in enumerate(CONTROLS))
    return (
        struct.pack("<BB", alpha0, alpha1)
        + alpha.to_bytes(6, "little")
        + struct.pack("<HHI", color0, color1, bits)
    )


def _dxt5_alpha(code: int, alpha0: int, alpha1: int) -> int:
    if code == 0:
        return alpha0
    if code == 1:
        return alpha1
    if alpha0 > alpha1:
        return ((8 - code) * alpha0 + (code - 1) * alpha1) // 7
    if code == 6:
        return 0
    if code == 7:
        return 255
    return ((6 - code) * alpha0 + (code - 1) * alpha1) // 5


DXT_BLOCK_COLORS = [(0xF800, 0x001F), (0x07E0, 0x001F)]


@pytest.mark.parametrize("size", [(1, 2), (2, 4), (3, 5)])
@pytest.mark.parametrize("alpha", [True, False])
def test_load_blp2_dxt3_narrow(size: tuple[int, int], alpha: bool) -> None:
    width, height = size
    blocks_per_row = (width + 3) // 4
    blocks_per_column = (height + 3) // 4
    nibbles = [
        [i % 16 for i in range(16)],
        [15 - i for i in range(16)],
    ]
    blocks = b"".join(
        _dxt3_block(
            nibbles[by % len(nibbles)],
            *DXT_BLOCK_COLORS[by % len(DXT_BLOCK_COLORS)],
        )
        for by in range(blocks_per_column)
        for _ in range(blocks_per_row)
    )

    with Image.open(
        _blp2_dxt(width, height, BlpImagePlugin.AlphaEncoding.DXT3, alpha, blocks)
    ) as im:
        im.load()
        assert im.mode == "RGBA" if alpha else "RGB"
        pixelsize = 4 if alpha else 3
        assert len(im.tobytes()) == width * height * pixelsize
        for y in range(height):
            block_nibbles = nibbles[(y // 4) % len(nibbles)]
            color0, color1 = DXT_BLOCK_COLORS[(y // 4) % len(DXT_BLOCK_COLORS)]
            for x in range(width):
                index = (y % 4) * 4 + (x % 4)
                expected: tuple[int, ...] = _dxt1_color(color0, color1, CONTROLS[index])
                if alpha:
                    expected += (block_nibbles[index] * 17,)
                assert im.getpixel((x, y)) == expected


@pytest.mark.parametrize("size", [(1, 2), (2, 4), (3, 5)])
@pytest.mark.parametrize("alpha", [True, False])
def test_load_blp2_dxt5_narrow(size: tuple[int, int], alpha: bool) -> None:
    width, height = size
    blocks_per_row = (width + 3) // 4
    blocks_per_column = (height + 3) // 4
    codes = [
        [i % 8 for i in range(16)],
        [7 - i % 8 for i in range(16)],
    ]
    alphas = [(0xFF, 0x00), (0x00, 0xFF)]
    blocks = b"".join(
        _dxt5_block(
            codes[by % len(codes)],
            *alphas[by % len(alphas)],
            *DXT_BLOCK_COLORS[by % len(DXT_BLOCK_COLORS)],
        )
        for by in range(blocks_per_column)
        for _ in range(blocks_per_row)
    )

    with Image.open(
        _blp2_dxt(width, height, BlpImagePlugin.AlphaEncoding.DXT5, alpha, blocks)
    ) as im:
        im.load()
        assert im.mode == "RGBA" if alpha else "RGB"
        pixelsize = 4 if alpha else 3
        assert len(im.tobytes()) == width * height * pixelsize
        for y in range(height):
            block_codes = codes[(y // 4) % len(codes)]
            alpha0, alpha1 = alphas[(y // 4) % len(alphas)]
            color0, color1 = DXT_BLOCK_COLORS[(y // 4) % len(DXT_BLOCK_COLORS)]
            for x in range(width):
                index = (y % 4) * 4 + (x % 4)
                expected: tuple[int, ...] = _dxt1_color(color0, color1, CONTROLS[index])
                if alpha:
                    expected += (_dxt5_alpha(block_codes[index], alpha0, alpha1),)
                assert im.getpixel((x, y)) == expected


@pytest.mark.parametrize(
    "alpha_encoding,blocksize",
    [
        (BlpImagePlugin.AlphaEncoding.DXT1, 8),
        (BlpImagePlugin.AlphaEncoding.DXT3, 16),
        (BlpImagePlugin.AlphaEncoding.DXT5, 16),
    ],
)
def test_load_blp2_dxt_truncated(
    alpha_encoding: BlpImagePlugin.AlphaEncoding, blocksize: int
) -> None:
    # 3x5 needs two block rows, but only one is provided
    blocks = bytes(blocksize)

    with Image.open(_blp2_dxt(3, 5, alpha_encoding, True, blocks)) as im:
        with pytest.raises(OSError):
            im.load()


@pytest.mark.parametrize(
    "alpha_encoding,blocksize",
    [
        (BlpImagePlugin.AlphaEncoding.DXT1, 8),
        (BlpImagePlugin.AlphaEncoding.DXT3, 16),
        (BlpImagePlugin.AlphaEncoding.DXT5, 16),
    ],
)
def test_decode_dxt_partial_block(
    alpha_encoding: BlpImagePlugin.AlphaEncoding, blocksize: int
) -> None:
    decode: dict[BlpImagePlugin.AlphaEncoding, Callable[[bytes], object]] = {
        BlpImagePlugin.AlphaEncoding.DXT1: BlpImagePlugin.decode_dxt1,
        BlpImagePlugin.AlphaEncoding.DXT3: BlpImagePlugin.decode_dxt3,
        BlpImagePlugin.AlphaEncoding.DXT5: BlpImagePlugin.decode_dxt5,
    }
    for length in (1, blocksize - 1, blocksize + 1):
        with pytest.raises(ValueError, match="whole number of blocks"):
            decode[alpha_encoding](bytes(length))


@pytest.mark.parametrize("size", [(0, 4), (4, 0), (0, 0)])
def test_load_blp2_dxt_invalid_size(size: tuple[int, int]) -> None:
    with pytest.raises(BlpImagePlugin.BLPFormatError):
        Image.open(_blp2_dxt(*size, BlpImagePlugin.AlphaEncoding.DXT1, False, bytes(8)))


def test_invalid_file() -> None:
    invalid_file = "Tests/images/flower.jpg"

    with pytest.raises(BlpImagePlugin.BLPFormatError):
        BlpImagePlugin.BlpImageFile(invalid_file)


def test_save(tmp_path: Path) -> None:
    f = tmp_path / "temp.blp"

    for version in ("BLP1", "BLP2"):
        im = hopper("P")
        im.save(f, blp_version=version)

        assert_image_equal_tofile(im.convert("RGB"), f)

        with Image.open("Tests/images/transparent.png") as im:
            f = tmp_path / "temp.blp"
            im.convert("P").save(f, blp_version=version)

            assert_image_similar_tofile(im, f, 8)

    im = hopper()
    with pytest.raises(ValueError, match="Unsupported BLP image mode"):
        im.save(f)


@pytest.mark.parametrize(
    "test_file",
    [
        "Tests/images/timeout-060745d3f534ad6e4128c51d336ea5489182c69d.blp",
        "Tests/images/timeout-31c8f86233ea728339c6e586be7af661a09b5b98.blp",
        "Tests/images/timeout-60d8b7c8469d59fc9ffff6b3a3dc0faeae6ea8ee.blp",
        "Tests/images/timeout-8073b430977660cdd48d96f6406ddfd4114e69c7.blp",
        "Tests/images/timeout-bba4f2e026b5786529370e5dfe9a11b1bf991f07.blp",
        "Tests/images/timeout-d6ec061c4afdef39d3edf6da8927240bb07fe9b7.blp",
        "Tests/images/timeout-ef9112a065e7183fa7faa2e18929b03e44ee16bf.blp",
    ],
)
def test_crashes(test_file: str) -> None:
    with open(test_file, "rb") as f:
        with Image.open(f) as im:
            with pytest.raises(OSError):
                im.load()
