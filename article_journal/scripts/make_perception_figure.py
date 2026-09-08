"""Compose the onboard gate-perception figure from committed HoloOcean captures.

Post-processing only. This script does **not** launch HoloOcean and does not
modify any artifact: it reads the three committed diagnostic screenshots under
``artifacts_gen2/visual_collection_smoke/`` and writes one PDF into
``article_journal/figures/generated/``.

The source captures were produced by the interactive perception viewer, whose
on-screen badges are localized (Italian) and which draws a keyboard-help bar at
the bottom. Both are viewer chrome rather than measurements, so this script
crops the camera panel and repaints the state badge in English. Every geometric
overlay drawn by the detector -- tracked aperture ROI, the four ordered corner
markers, the image-centre crosshair, the estimated gate centre and the image
error vector -- is left untouched, and every number quoted in the caption is
copied verbatim from the corresponding on-screen readout.

Usage:
    python article_journal/scripts/make_perception_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "artifacts_gen2" / "visual_collection_smoke"
OUT = ROOT / "article_journal" / "figures" / "generated"

# Camera panel of the three-view viewer: columns [0, 640), rows [0, 445).
# Column 640 is the first column of the second view (verified numerically);
# row 445 is above the localized panel footer.
PANEL = (0, 0, 640, 445)
# Bounding box of the localized state badge inside the cropped panel.
BADGE = (8, 16, 392, 84)
INK = (24, 34, 46)

# (source screenshot, English state badge, subcaption)
PANELS = [
    (
        SMOKE / "20260907_191258_mixed_endurance_seed67002" / "screenshots"
        / "00000_detection_without_corners.png",
        [("PROPOSAL", (120, 230, 140)),
         ("aperture: 4 corners unavailable", (235, 120, 110))],
        "(a) No valid aperture quadrilateral",
    ),
    (
        SMOKE / "20260907_191120_horseshoe_bay_seed67000" / "screenshots"
        / "00001_corners_B01.png",
        [("LOCK", (120, 230, 140)),
         ("frontal   yaw = +2.0 deg   conf = 0.51", (150, 235, 130))],
        "(b) Four ordered corners; target locked to B01",
    ),
    (
        SMOKE / "20260907_191216_vertical_serpent_seed67001" / "screenshots"
        / "00035_multi_candidate_lock.png",
        [("LOCK", (120, 230, 140)),
         ("frontal   yaw = -7.3 deg   conf = 0.28", (150, 235, 130))],
        "(c) Near-field passage; two candidates",
    ),
]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("consola.ttf", "cour.ttf", "arial.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_panel(path: Path, badge_lines) -> Image.Image:
    image = Image.open(path).convert("RGB").crop(PANEL)
    draw = ImageDraw.Draw(image)
    draw.rectangle(BADGE, fill=INK)
    x0, y0, _, _ = BADGE
    for index, (text, colour) in enumerate(badge_lines):
        size = 24 if index == 0 else 17
        draw.text((x0 + 8, y0 + 6 + index * 30), text, fill=colour, font=_font(size))
    return image


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p, _, _ in PANELS if not p.exists()]
    if missing:
        raise SystemExit(f"missing source capture(s): {missing}")

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 1.95))
    for axis, (path, badge, caption) in zip(axes, PANELS):
        axis.imshow(build_panel(path, badge))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#12263a")
            spine.set_linewidth(0.6)
        axis.set_title(caption, fontsize=6.8, color="#12263a", pad=3.0)
    figure.subplots_adjust(left=0.004, right=0.996, top=0.87, bottom=0.01, wspace=0.03)
    target = OUT / "gate_perception_panels.pdf"
    figure.savefig(target, dpi=400)
    plt.close(figure)

    provenance = {
        "figure": "gate_perception_panels.pdf",
        "script": "article_journal/scripts/make_perception_figure.py",
        "operation": "crop + English state badge only; detector overlays untouched",
        "sources": [str(p.relative_to(ROOT)).replace("\\", "/") for p, _, _ in PANELS],
        "source_capture_report": "artifacts_gen2/visual_collection_smoke/visual_smoke_report.json",
        "adapter": "holoocean",
        "seeds": [67002, 67000, 67001],
        "tracks": ["mixed_endurance", "vertical_serpent", "horseshoe_bay"],
        "holoocean_launched_by_this_script": False,
    }
    (OUT / "gate_perception_panels.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
