from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .imageio import open_pil
from .pipeline import IdentifyResult

KNOWN = (34, 170, 90)
UNKNOWN = (220, 60, 60)
WARN = (235, 150, 20)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def annotate(source, result: IdentifyResult) -> Image.Image:
    """Draw boxes and labels on the (EXIF-corrected) photo. Used by the CLI and for exports."""
    img = open_pil(source)
    draw = ImageDraw.Draw(img)
    unit = max(img.size) / 1000
    line = max(2, round(3 * unit))
    font = _font(max(12, round(20 * unit)))

    for f in result.faces:
        colour = KNOWN if f.status == "known" else UNKNOWN
        x1, y1, x2, y2 = f.bbox
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line)
        label = f"{f.index}. {f.name if f.status == 'known' else 'Unknown'}"
        if f.low_quality:
            label += " (?)"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, line)
        ty = y1 - th - 2 * pad
        if ty < 0:
            ty = y2
        tx = max(0, min(x1, img.width - (tw + 2 * pad)))   # keep the label inside the photo
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=colour)
        draw.text((tx + pad, ty + pad - tb[1]), label, fill="white", font=font)
    return img
