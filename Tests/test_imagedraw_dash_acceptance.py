"""Pixel-level acceptance tests for dashed ImageDraw drawing.

These tests use tiny fixed images so that dash lengths, gaps, starting
phase and phase continuity across corners are checked against exact
pixel results rather than image hashes.
"""

from __future__ import annotations

import pytest

from PIL import Image, ImageDraw


def _rows(im: Image.Image) -> list[str]:
    px = im.load()
    return [
        "".join("#" if px[x, y] else "." for x in range(im.width))
        for y in range(im.height)
    ]


def _blank(w: int, h: int) -> ImageDraw.ImageDraw:
    im = Image.new("L", (w, h), 0)
    return ImageDraw.Draw(im), im


@pytest.mark.parametrize(
    "dash",
    [
        (),
        (0,),
        (-1,),
        (0, -1),
        (1, 0),
        (-4, 2),
        (1.5, 2),
        (1, "2"),
        ("a", "b"),
        [],
        5,
        "ab",
    ],
)
@pytest.mark.parametrize("method", ["line", "polygon", "rectangle"])
def test_dash_invalid_parameters(dash: object, method: str) -> None:
    draw, _ = _blank(20, 20)
    with pytest.raises(ValueError, match="dash must be a non-empty tuple"):
        if method == "line":
            draw.line([(0, 0), (10, 10)], dash=dash)  # type: ignore[arg-type]
        elif method == "polygon":
            draw.polygon([(0, 0), (10, 0), (0, 10)], dash=dash)  # type: ignore[arg-type]
        else:
            draw.rectangle([0, 0, 10, 10], dash=dash)  # type: ignore[arg-type]


def test_dash_list_is_accepted() -> None:
    a = Image.new("L", (12, 3), 0)
    ImageDraw.Draw(a).line([(0, 1), (11, 1)], fill=255, dash=[3, 2])
    b = Image.new("L", (12, 3), 0)
    ImageDraw.Draw(b).line([(0, 1), (11, 1)], fill=255, dash=(3, 2))
    assert a.tobytes() == b.tobytes()


def test_dash_polyline_fixed_bitmap() -> None:
    draw, im = _blank(16, 3)
    draw.line([(1, 1), (14, 1)], fill=255, dash=(4, 2))

    rows = _rows(im)
    assert rows[0] == "." * 16
    # 4 on, 2 off, 4 on, 2 off, then the terminal coordinate pixel.
    assert rows[1] == ".####..####..##."
    assert rows[2] == "." * 16


def test_dash_none_matches_omitted() -> None:
    a = Image.new("L", (40, 30), 0)
    ImageDraw.Draw(a).line([(2, 2), (20, 26), (38, 4)], fill=255, width=3, dash=None)
    b = Image.new("L", (40, 30), 0)
    ImageDraw.Draw(b).line([(2, 2), (20, 26), (38, 4)], fill=255, width=3)
    assert a.tobytes() == b.tobytes()

    r1 = Image.new("RGB", (40, 40))
    ImageDraw.Draw(r1).rectangle([2, 2, 37, 37], outline="red", width=2, dash=None)
    r2 = Image.new("RGB", (40, 40))
    ImageDraw.Draw(r2).rectangle([2, 2, 37, 37], outline="red", width=2)
    assert r1.tobytes() == r2.tobytes()


def test_dash_width_and_joint_ignored() -> None:
    thin = Image.new("L", (30, 12), 0)
    ImageDraw.Draw(thin).line([(0, 6), (29, 6)], fill=255, width=1, dash=(4, 3))
    thick = Image.new("L", (30, 12), 0)
    ImageDraw.Draw(thick).line(
        [(0, 6), (29, 6)], fill=255, width=6, joint="curve", dash=(4, 3)
    )
    assert thin.tobytes() == thick.tobytes()


def test_dash_cross_corner_phase_continues() -> None:
    # L shape (2,2) -> (8,2) -> (8,8), dash=(3, 2), period 5.
    draw, im = _blank(12, 11)
    draw.line([(2, 2), (8, 2), (8, 8)], fill=255, dash=(3, 2))

    expected_on = {
        # horizontal run: indices 0..5
        (2, 2),
        (3, 2),
        (4, 2),
        (7, 2),
        # vertical run continues at global index 6, not at phase 0
        (8, 2),  # global 6 mod 5 == 1, on
        (8, 3),  # global 7 mod 5 == 2, on
        (8, 6),  # global 10 mod 5 == 0, on
        (8, 7),  # global 11 mod 5 == 1, on
        (8, 8),  # terminal coordinate pixel
    }
    for y in range(im.height):
        for x in range(im.width):
            assert im.getpixel((x, y)) == (255 if (x, y) in expected_on else 0), (
                x,
                y,
            )

    # Two independent calls restart the phase at the new segment start.
    separate, im2 = _blank(12, 11)
    separate.line([(2, 2), (8, 2)], fill=255, dash=(3, 2))
    separate.line([(8, 2), (8, 8)], fill=255, dash=(3, 2))
    # Restarted phase makes local index 2 (8, 4) lit; continuous phase leaves
    # it blank.
    assert im2.getpixel((8, 4)) == 255
    assert im.getpixel((8, 4)) == 0


def test_dash_rectangle_matches_closed_polyline() -> None:
    box = [2, 3, 21, 17]
    rect = Image.new("L", (24, 22), 0)
    ImageDraw.Draw(rect).rectangle(box, outline=255, dash=(5, 2))

    poly = Image.new("L", (24, 22), 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(poly).line(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
        fill=255,
        dash=(5, 2),
    )
    assert rect.tobytes() == poly.tobytes()

    # Nested coordinate form renders identically to the flat form.
    nested = Image.new("L", (24, 22), 0)
    ImageDraw.Draw(nested).rectangle([(x0, y0), (x1, y1)], outline=255, dash=(5, 2))
    assert nested.tobytes() == rect.tobytes()


def test_dash_rectangle_corner_phase_pixels() -> None:
    # Rectangle [1,1,9,7], dash=(3,2). Phase accumulates around the corners.
    draw, im = _blank(11, 9)
    draw.rectangle([1, 1, 9, 7], outline=255, dash=(3, 2))

    on = {
        (1, 1),  # top-left start (global 0)
        (2, 1),
        (3, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 3),  # right edge, global 10 -> on
        (9, 4),
        (9, 5),
        (8, 7),  # bottom edge
        (7, 7),
        (6, 7),
        (3, 7),
        (2, 7),
        (1, 7),  # bottom-left, global 22 mod 5 == 2 -> on
        (1, 4),  # left edge, global 25 mod 5 == 0 -> on
        (1, 3),
        (1, 2),  # global 27 mod 5 == 2 -> on
    }
    off = {
        (4, 1),
        (5, 1),
        (9, 1),  # top-right corner lands in a gap
        (9, 2),
        (9, 6),
        (9, 7),  # bottom-right corner lands in a gap
        (5, 7),
        (4, 7),
        (1, 6),
        (1, 5),
    }
    for p in on:
        assert im.getpixel(p) == 255, f"expected on at {p}"
    for p in off:
        assert im.getpixel(p) == 0, f"expected off at {p}"


@pytest.mark.parametrize("dash", [(4, 2), (4, 2, 1, 2), (1, 2, 3)])
def test_dash_polygon_equals_closed_polyline(dash: tuple[int, ...]) -> None:
    triangle = [(3, 3), (20, 6), (6, 20)]
    poly_im = Image.new("L", (24, 24), 0)
    ImageDraw.Draw(poly_im).polygon(triangle, outline=255, dash=dash)

    line_im = Image.new("L", (24, 24), 0)
    ImageDraw.Draw(line_im).line(triangle + [triangle[0]], fill=255, dash=dash)
    assert poly_im.tobytes() == line_im.tobytes()


def test_dash_polygon_fill_and_outline() -> None:
    im = Image.new("RGB", (30, 30), "white")
    d = ImageDraw.Draw(im)
    d.polygon(
        [(5, 5), (25, 5), (5, 25)],
        fill="red",
        outline="green",
        dash=(4, 2),
    )
    # Fill is present in the interior.
    assert im.getpixel((8, 8)) == (255, 0, 0)
    # Dashed outline produced green pixels along the top edge, while the
    # gaps reveal the red fill drawn underneath rather than the background.
    top = [im.getpixel((x, 5)) for x in range(5, 26)]
    assert (0, 128, 0) in top
    assert (255, 0, 0) in top


@pytest.mark.parametrize(
    "operation",
    [
        lambda d: d.line([(4, 4), (4, 4)], fill=255, dash=(2, 2)),
        lambda d: d.line([(1, 1), (1, 1), (8, 8), (8, 8)], fill=255, dash=(3, 2)),
        lambda d: d.polygon([(4, 4), (4, 4), (6, 4), (6, 6)], outline=255, dash=(2, 1)),
        lambda d: d.rectangle([5, 5, 5, 5], outline=255, dash=(2, 1)),
        lambda d: d.rectangle([2, 2, 2, 8], outline=255, dash=(3, 2)),
        lambda d: d.rectangle([8, 8, 2, 2], outline=255, dash=(3, 2)),
    ],
)
def test_dash_zero_length_and_degenerate(operation) -> None:
    _, im = _blank(11, 11)
    operation(ImageDraw.Draw(im))  # must not raise or hang


def test_dash_empty_polygon_still_raises_type_error() -> None:
    draw, _ = _blank(8, 8)
    with pytest.raises(TypeError, match="at least 2 coordinates"):
        draw.polygon([], outline=255, dash=(2, 2))


def test_dash_moderate_clipping_is_bounded() -> None:
    im = Image.new("L", (20, 20), 0)
    d = ImageDraw.Draw(im)
    d.line([(-20, 10), (40, 10)], fill=255, dash=(6, 4))
    d.rectangle([-8, 2, 15, 17], outline=255, dash=(4, 3))
    # Drawing stays inside the image and produces visible pixels.
    assert im.getbbox() is not None
    bbox = im.getbbox()
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= 20
    assert bbox[3] <= 20


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (2**31 - 1, 2**31 - 1)],
        [(-(2**31), -(2**31)), (2**31 - 1, 2**31 - 1)],
    ],
)
def test_dash_extreme_coordinates_do_not_hang(points) -> None:
    im = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(im).line(points, fill=255, dash=(5, 3))
    # Reaching this point quickly is the assertion; visible output is bounded.
    if im.getbbox() is not None:
        assert 0 <= im.getbbox()[0] <= 10
