#!/usr/bin/env python3
"""Draw the PhosphoMAX app icon and write a multi-resolution .icns.

macOS builds .icns with `iconutil`, which only exists on macOS. The format is
simple enough to write directly: a 'icns' magic, a big-endian total length,
then one chunk per size — a 4-byte type code, a big-endian chunk length, and
a PNG payload. Doing it here means the finished bundle needs no build step on
the Mac.
"""

import math
import os
import struct
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── palette ───────────────────────────────────────────────────────────────────
BG_TOP    = (26, 62, 122)     # deep blue
BG_BOT    = (14, 148, 152)    # teal
CURVE     = (255, 255, 255)
POINT     = (255, 255, 255)
POINT_RIM = (26, 62, 122)
GRID      = (255, 255, 255, 30)

S = 1024                      # master size; everything scales from this
SS = 3                        # supersampling factor for the plot artwork


def _vertical_gradient(size, top, bot):
    grad = Image.new("RGB", (1, size), top)
    px = grad.load()
    for y in range(size):
        t = y / max(1, size - 1)
        # ease-in-out so the midtone sits slightly high, which reads better
        # once the icon is scaled down to 32 px
        t = t * t * (3 - 2 * t)
        px[0, y] = tuple(round(a + (b - a) * t) for a, b in zip(top, bot))
    return grad.resize((size, size), Image.BILINEAR)


def _squircle_mask(size, radius_frac=0.2237):
    """Apple's rounded-rectangle proportion, drawn at 4x then downsampled so
    the corners are properly antialiased."""
    ss = size * 4
    m = Image.new("L", (ss, ss), 0)
    d = ImageDraw.Draw(m)
    inset = round(ss * 0.055)
    d.rounded_rectangle([inset, inset, ss - inset - 1, ss - inset - 1],
                        radius=round(ss * radius_frac), fill=255)
    return m.resize((size, size), Image.LANCZOS)


def _sigmoid(t, x0=0.46, k=7.2):
    """The dose-response S-curve as the app actually plots it: response
    against log concentration. Reads far better as a silhouette than the
    hyperbola does, and it is what most of these panels look like."""
    return 1.0 / (1.0 + math.exp(-k * (t - x0)))


def _stroke(draw, pts, colour, width):
    """Round-capped, round-joined polyline.

    PIL's joint="curve" spits out visible spikes on a densely sampled path,
    so the stroke is stamped as overlapping discs instead — slower, but the
    edge is clean at every size.
    """
    r = width / 2.0
    draw.line(pts, fill=colour, width=int(round(width)))
    for (cx, cy) in pts:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)


def _detail_for(target):
    """How much to draw, given the size the icon will actually be seen at.

    A 16 px tile cannot hold a grid and six ringed markers — they collapse
    into noise. Small sizes get a bolder curve and nothing else, which is
    what makes an icon still readable in a Finder list.
    """
    if target <= 32:
        return dict(grid=0, points=0, lw=0.075, axes=0.020, glow=False)
    if target <= 64:
        return dict(grid=2, points=4, lw=0.058, axes=0.017, glow=False,
                    pr=0.040, rim=0.014)
    if target <= 128:
        return dict(grid=3, points=5, lw=0.048, axes=0.015, glow=True,
                    pr=0.036, rim=0.012)
    return dict(grid=3, points=5, lw=0.040, axes=0.014, glow=True,
                pr=0.033, rim=0.010)


def draw_icon(size=S, target=None):
    """Artwork is composed at SS× and downsampled, so every edge is smooth.
    *target* is the size it will be displayed at, which sets the detail."""
    cfg = _detail_for(target if target is not None else size)
    z = size * SS
    base = _vertical_gradient(z, BG_TOP, BG_BOT).convert("RGBA")
    d = ImageDraw.Draw(base, "RGBA")

    # plot area
    x0, x1 = z * 0.205, z * 0.885
    y0, y1 = z * 0.215, z * 0.735
    w, h = x1 - x0, y1 - y0

    # faint grid
    n = cfg["grid"]
    if n:
        gw = max(1, round(z * 0.0035))
        for i in range(1, n + 1):
            yy = y0 + h * i / (n + 1)
            d.line([(x0, yy), (x1, yy)], fill=GRID, width=gw)
            xx = x0 + w * i / (n + 1)
            d.line([(xx, y0), (xx, y1)], fill=GRID, width=gw)

    # axes
    d.line([(x0, y0), (x0, y1), (x1, y1)],
           fill=(255, 255, 255, 200), width=max(2, round(z * cfg["axes"])))

    # the curve, inset from the axes so it never merges with them
    pad = 0.05
    def _at(t):
        return (x0 + (pad + t * (1 - 2 * pad)) * w,
                y1 - (0.06 + _sigmoid(t) * 0.88) * h)

    pts = [_at(i / 400) for i in range(401)]
    lw = z * cfg["lw"]
    if cfg["glow"]:
        glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
        _stroke(ImageDraw.Draw(glow, "RGBA"), pts, (255, 255, 255, 80), lw * 2.4)
        base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(z * 0.014)))
    _stroke(d, pts, CURVE, lw)

    # data points sitting on the curve
    npts = cfg["points"]
    if npts:
        r = z * cfg["pr"]
        rim = max(2, round(z * cfg["rim"]))
        for i in range(npts):
            t = 0.10 + i * (0.83 / (npts - 1))
            cx, cy = _at(t)
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=POINT, outline=POINT_RIM, width=rim)

    return base.resize((size, size), Image.LANCZOS)


def _font(px):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return None


def add_wordmark(img, text="pKd"):
    """Only drawn on the larger sizes — below 128 px it turns to mud."""
    size = img.width
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(round(size * 0.105))
    if f is None:
        return img
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - tw) / 2 - bbox[0], size * 0.845 - th / 2 - bbox[1]),
           text, font=f, fill=(255, 255, 255, 200))
    return img


_MASTERS = {}


def render(size, wordmark=True):
    """Render at *size*, with the detail level appropriate to that size."""
    key = _detail_for(size)["lw"]        # one master per detail tier
    if key not in _MASTERS:
        _MASTERS[key] = draw_icon(S, target=size)
    icon = _MASTERS[key].copy()
    if wordmark and size >= 256:
        icon = add_wordmark(icon)
    icon = icon.resize((size, size), Image.LANCZOS)
    mask = _squircle_mask(size)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(icon, (0, 0), mask)
    return out


# ── ICNS container ────────────────────────────────────────────────────────────
# Type codes and the pixel size each one means. The 'ic' PNG-based codes are
# understood by every macOS version that matters.
ICNS_TYPES = [
    (b"icp4",   16), (b"icp5",   32), (b"icp6",   64),
    (b"ic07",  128), (b"ic08",  256), (b"ic09",  512), (b"ic10", 1024),
    (b"ic11",   32),   # 16pt @2x
    (b"ic12",   64),   # 32pt @2x
    (b"ic13",  512),   # 256pt @2x
    (b"ic14", 1024),   # 512pt @2x
]


def build_icns(path):
    chunks = []
    cache = {}
    for code, px in ICNS_TYPES:
        if px not in cache:
            buf = _png_bytes(render(px))
            cache[px] = buf
        data = cache[px]
        chunks.append(code + struct.pack(">I", len(data) + 8) + data)
    body = b"".join(chunks)
    blob = b"icns" + struct.pack(">I", len(body) + 8) + body
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def _png_bytes(img):
    import io
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    icns = os.path.join(out_dir, "PhosphoMAX.icns")
    n = build_icns(icns)
    print(f"{icns}  ({n/1024:.0f} kB)")
    # A PNG preview alongside, handy for checking and for reuse elsewhere.
    prev = os.path.join(out_dir, "PhosphoMAX_icon_preview.png")
    render(512).save(prev)
    print(f"{prev}")
