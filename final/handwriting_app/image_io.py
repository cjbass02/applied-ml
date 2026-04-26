"""Convert raw mouse strokes into EMNIST-compatible 28x28 PNGs on disk.

The save pipeline mirrors the orientation used by EMNIST raw bytes so that the
notebook's ``emnist_transform`` (``ToTensor -> rotate(-90) -> hflip``) re-uprights
our personal images identically to EMNIST samples.

Concretely, given an upright drawing ``D`` we want
``hflip(rotate(-90, stored)) == D``, which solves to
``stored = rotate(+90, hflip(D))``. That is what :func:`save_drawing` does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageOps

CANVAS_SIZE = 280
EMNIST_SIZE = 28
DEFAULT_PEN_RADIUS = 10
PADDING_FRACTION = 0.18


Stroke = Iterable[Tuple[float, float]]


def render_strokes_to_image(
    strokes: Iterable[Stroke],
    canvas_size: int = CANVAS_SIZE,
    pen_radius: int = DEFAULT_PEN_RADIUS,
) -> Image.Image:
    """Rasterize a list of stroke point sequences into a grayscale image.

    Output is white-on-black (matching EMNIST upright orientation), which is
    what callers will then crop, resize, and reorient.
    """
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        points = list(stroke)
        if not points:
            continue
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse(
                (x - pen_radius, y - pen_radius, x + pen_radius, y + pen_radius),
                fill=255,
            )
            continue
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            draw.line((x0, y0, x1, y1), fill=255, width=pen_radius * 2)
            draw.ellipse(
                (x1 - pen_radius, y1 - pen_radius, x1 + pen_radius, y1 + pen_radius),
                fill=255,
            )
        x0, y0 = points[0]
        draw.ellipse(
            (x0 - pen_radius, y0 - pen_radius, x0 + pen_radius, y0 + pen_radius),
            fill=255,
        )

    return img


def crop_center_resize(
    img: Image.Image,
    out_size: int = EMNIST_SIZE,
    padding_fraction: float = PADDING_FRACTION,
) -> Image.Image:
    """Crop to inked bbox, square-pad, then resize to ``out_size``.

    Mirrors the standard MNIST/EMNIST centering: the glyph is tightly cropped,
    centered on a square, padded by a small margin, and downsampled with
    antialiasing. Returns a grayscale image of size ``(out_size, out_size)``.
    """
    bbox = img.getbbox()
    if bbox is None:
        return Image.new("L", (out_size, out_size), color=0)

    cropped = img.crop(bbox)
    w, h = cropped.size
    side = max(w, h)
    pad = int(side * padding_fraction)
    canvas_side = side + 2 * pad

    canvas = Image.new("L", (canvas_side, canvas_side), color=0)
    offset_x = (canvas_side - w) // 2
    offset_y = (canvas_side - h) // 2
    canvas.paste(cropped, (offset_x, offset_y))

    return canvas.resize((out_size, out_size), Image.LANCZOS)


def to_emnist_storage_orientation(img: Image.Image) -> Image.Image:
    """Take an upright glyph and return the EMNIST raw-byte orientation.

    Forward pipeline (transform): ``ToTensor -> rotate(-90) -> hflip``.
    Inverse (storage):            ``hflip -> rotate(+90)``.
    """
    return ImageOps.mirror(img).rotate(90, expand=False)


def save_drawing(
    strokes: Iterable[Stroke],
    out_dir: Path,
    *,
    class_name: str,
    canvas_size: int = CANVAS_SIZE,
    pen_radius: int = DEFAULT_PEN_RADIUS,
) -> Path:
    """Render, normalize, reorient, and save strokes for one sample.

    The filename is ``{timestamp}_{class}.png`` to make resuming and merging
    sessions painless. Returns the path that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    upright = render_strokes_to_image(strokes, canvas_size=canvas_size, pen_radius=pen_radius)
    normalized = crop_center_resize(upright)
    storage = to_emnist_storage_orientation(normalized)

    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    out_path = out_dir / f"{ts}_{class_name}.png"
    storage.save(out_path)
    return out_path
