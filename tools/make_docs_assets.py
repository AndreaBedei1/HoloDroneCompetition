"""Build the documentation images from real repository material.

Two outputs, both written to ``docs/assets/``:

* ``mra_banner.png``    -- the README hero. Composites the committed HoloOcean
                           course render (``article_journal/figures/captures/``)
                           with the wordmark and the three official circuit
                           outlines drawn from the track JSON files.
* ``official_tracks.png`` -- top-down outlines of the three official circuits,
                           drawn from the same track JSON files.

Nothing here invents content: the photograph is a committed HoloOcean capture
and every gate position comes from ``marine_race_arena/tracks/``. No simulator
is launched.

Usage:
    python tools/make_docs_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "marine_race_arena" / "tracks"
RENDER = ROOT / "article_journal" / "figures" / "captures" / "holoocean_course_render.png"
OUT = ROOT / "docs" / "assets"

FONTS = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"
BOLD = FONTS / "DejaVuSans-Bold.ttf"
REGULAR = FONTS / "DejaVuSans.ttf"

# Palette: the deep-water navy of the render plus the gate colour used by the
# official tracks (#00ff88 in every gate definition).
NAVY = (9, 22, 38)
NAVY_SOFT = (16, 38, 61)
GATE = (0, 255, 136)
INK = (231, 240, 248)
MUTED = (139, 166, 189)

OFFICIAL = [
    ("marine_race_horseshoe_bay.json", "HORSESHOE BAY"),
    ("marine_race_vertical_serpent.json", "VERTICAL SERPENT"),
    ("marine_race_mixed_endurance.json", "MIXED ENDURANCE"),
]


def load_track(name: str) -> dict:
    data = json.loads((TRACKS / name).read_text(encoding="utf-8"))
    by_id = {gate["id"]: gate for gate in data["gates"]}
    gates = [by_id[gid] for gid in data["track"]["gate_sequence"]]
    return {
        "gates": gates,
        "start": data["start"]["position"],
        "length": data["track"]["declared_length_m"],
        "count": len(gates),
    }


def fit(points, box, pad):
    """Map (x, y) world points into a pixel box, preserving aspect."""
    x0, y0, x1, y1 = box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    scale = min((x1 - x0 - 2 * pad) / span_x, (y1 - y0 - 2 * pad) / span_y)
    off_x = x0 + ((x1 - x0) - span_x * scale) / 2 - min(xs) * scale
    off_y = y0 + ((y1 - y0) - span_y * scale) / 2 - min(ys) * scale
    # screen y grows downward; flip so the layout reads like a map
    return [(off_x + p[0] * scale, (y1 + y0) - (off_y + p[1] * scale)) for p in points]


def draw_course(draw, track, box, *, pad, line, dot, colour=GATE, trail=(60, 96, 126)):
    points = [track["start"]] + [g["position"] for g in track["gates"]]
    placed = fit(points, box, pad)
    draw.line(placed, fill=trail, width=line, joint="curve")
    start = placed[0]
    draw.ellipse([start[0] - dot, start[1] - dot, start[0] + dot, start[1] + dot],
                 outline=INK, width=max(1, line))
    for point in placed[1:]:
        draw.ellipse([point[0] - dot, point[1] - dot, point[0] + dot, point[1] + dot],
                     fill=colour)
    finish = placed[-1]
    ring = dot * 2.4
    draw.ellipse([finish[0] - ring, finish[1] - ring, finish[0] + ring, finish[1] + ring],
                 outline=colour, width=max(1, line))


def draw_depth_profile(draw, track, box, *, line=3):
    """A small z-versus-gate ribbon: the top-down view alone hides the depth
    changes that make Vertical Serpent and Mixed Endurance three-dimensional."""
    x0, y0, x1, y1 = box
    depths = [gate["position"][2] for gate in track["gates"]]
    lo, hi = min(depths), max(depths)
    # a flat course must stay visibly flat, so never normalise a tiny span up
    span = max(hi - lo, 1.0)
    mid = (hi + lo) / 2
    step = (x1 - x0) / max(len(depths) - 1, 1)
    points = [(x0 + i * step, (y0 + y1) / 2 + ((mid - d) / span) * (y1 - y0) * 0.5)
              for i, d in enumerate(depths)]
    draw.line([(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)], fill=(31, 58, 84), width=1)
    draw.line(points, fill=(96, 176, 214), width=line, joint="curve")
    return hi, lo


def tracked_text(draw, xy, text, font, fill, spacing=0):
    """Draw text with manual letter spacing."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += draw.textlength(char, font=font) + spacing
    return x


def banner() -> Path:
    width, height = 2400, 760
    canvas = Image.new("RGB", (width, height), NAVY)

    # --- right side: the committed HoloOcean capture, cropped to a wide band --
    photo = Image.open(RENDER).convert("RGB")
    # a mild lift so the gate line and the vehicle read at banner scale
    photo = ImageEnhance.Brightness(photo).enhance(1.06)
    photo = ImageEnhance.Contrast(photo).enhance(1.14)
    target_w = int(width * 0.60)
    ratio = target_w / photo.width
    photo = photo.resize((target_w, int(photo.height * ratio)), Image.LANCZOS)
    top = max(0, (photo.height - height) // 2)
    photo = photo.crop((0, top, photo.width, min(photo.height, top + height)))
    if photo.height < height:                      # pad rather than distort
        padded = Image.new("RGB", (photo.width, height), NAVY)
        padded.paste(photo, (0, (height - photo.height) // 2))
        photo = padded
    canvas.paste(photo, (width - target_w, 0))

    # feather the photograph into the navy field on its left edge: a long,
    # gentle ramp so no seam is visible behind the wordmark
    fade = Image.new("L", (target_w, height), 255)
    fade_draw = ImageDraw.Draw(fade)
    blend = int(target_w * 0.62)
    for i in range(blend):
        fade_draw.line([(i, 0), (i, height)], fill=int(255 * (i / blend) ** 2.1))
    fade = fade.filter(ImageFilter.GaussianBlur(9))
    canvas.paste(Image.new("RGB", (target_w, height), NAVY),
                 (width - target_w, 0),
                 Image.eval(fade, lambda v: 255 - v))

    # a light overall scrim settles the water down behind the text column
    scrim = Image.new("L", (width, height), 0)
    scrim_draw = ImageDraw.Draw(scrim)
    for i in range(width):
        scrim_draw.line([(i, 0), (i, height)],
                        fill=int(90 * max(0.0, 1 - i / (width * 0.86))))
    canvas.paste(Image.new("RGB", (width, height), NAVY), (0, 0),
                 scrim.filter(ImageFilter.GaussianBlur(3)))

    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(BOLD), 116)
    lead = ImageFont.truetype(str(REGULAR), 44)
    small = ImageFont.truetype(str(BOLD), 26)

    left = 120
    tracked_text(draw, (left, 208), "MARINE RACE ARENA", title, INK, spacing=6)
    draw.line([(left + 4, 372), (left + 4 + 250, 372)], fill=GATE, width=6)
    draw.text((left, 416), "A configurable benchmark for autonomous", font=lead, fill=MUTED)
    draw.text((left, 470), "underwater gate racing.", font=lead, fill=MUTED)

    chips = ["HOLOOCEAN", "BLUEROV2", "ONBOARD-ONLY AUTONOMY", "INDEPENDENT REFEREE"]
    x = left
    for chip in chips:
        w = draw.textlength(chip, font=small)
        draw.rounded_rectangle([x, 566, x + w + 34, 614], radius=24,
                               outline=(46, 74, 100), width=2)
        draw.text((x + 17, 578), chip, font=small, fill=MUTED)
        x += w + 34 + 16

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mra_banner.png"
    canvas.save(path, optimize=True)
    print("wrote", path.relative_to(ROOT).as_posix(), canvas.size)
    return path


def tracks_strip() -> Path:
    width, height = 2100, 820
    canvas = Image.new("RGB", (width, height), NAVY)
    draw = ImageDraw.Draw(canvas)
    name_font = ImageFont.truetype(str(BOLD), 34)
    meta_font = ImageFont.truetype(str(REGULAR), 30)
    tiny_font = ImageFont.truetype(str(REGULAR), 22)

    panel_w = width // 3
    for index, (fname, label) in enumerate(OFFICIAL):
        track = load_track(fname)
        x0 = index * panel_w
        draw.rounded_rectangle([x0 + 48, 58, x0 + panel_w - 48, height - 58],
                               radius=26, fill=NAVY_SOFT)
        draw_course(draw, track, (x0 + 70, 150, x0 + panel_w - 70, height - 300),
                    pad=34, line=4, dot=8)
        tracked_text(draw, (x0 + 86, 96), label, name_font, INK, spacing=2)

        shallowest, deepest = draw_depth_profile(
            draw, track, (x0 + 96, height - 268, x0 + panel_w - 96, height - 188))
        draw.text((x0 + 96, height - 176),
                  "depth {:.1f}-{:.1f} m".format(-shallowest, -deepest),
                  font=tiny_font, fill=(112, 150, 180))
        draw.text((x0 + 86, height - 126),
                  "{} gates   {:.1f} m".format(track["count"], track["length"]),
                  font=meta_font, fill=GATE)

    draw.text((72, height - 46), "top-down gate sequence (above) and depth profile (below), "
                                 "drawn from marine_race_arena/tracks/",
              font=tiny_font, fill=(74, 104, 132))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "official_tracks.png"
    canvas.save(path, optimize=True)
    print("wrote", path.relative_to(ROOT).as_posix(), canvas.size)
    return path


def main() -> int:
    if not RENDER.is_file():
        raise SystemExit("missing course render: {}".format(RENDER))
    banner()
    tracks_strip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
