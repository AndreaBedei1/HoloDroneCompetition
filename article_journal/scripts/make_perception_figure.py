"""Compose the onboard gate-perception figure from committed HoloOcean captures.

Post-processing only. This script does **not** launch HoloOcean and does not
modify any artifact: it reads the three committed diagnostic screenshots under
``artifacts_gen2/visual_collection_smoke/`` and writes one PDF into
``article_journal/figures/generated/``.

The source captures were produced by the interactive perception viewer. This
script keeps only the camera panel crop; detector overlays are preserved
unchanged, while viewer chrome outside the crop is omitted.

Usage:
    python article_journal/scripts/make_perception_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "artifacts_gen2" / "visual_collection_smoke"
OUT = ROOT / "article_journal" / "figures" / "generated"

# Camera panel of the three-view viewer. The upper crop boundary excludes the
# viewer's state badge; the lower boundary is above the localized panel footer.
PANEL = (0, 90, 640, 445)

# Source screenshot order: Mixed Endurance, Horseshoe Bay, Vertical Serpent.
PANELS = [
    SMOKE / "20260907_191258_mixed_endurance_seed67002" / "screenshots"
    / "00000_detection_without_corners.png",
    SMOKE / "20260907_191120_horseshoe_bay_seed67000" / "screenshots"
    / "00001_corners_B01.png",
    SMOKE / "20260907_191216_vertical_serpent_seed67001" / "screenshots"
    / "00035_multi_candidate_lock.png",
]


def build_panel(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").crop(PANEL)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in PANELS if not p.exists()]
    if missing:
        raise SystemExit(f"missing source capture(s): {missing}")

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 1.70))
    for axis, path in zip(axes, PANELS):
        axis.imshow(build_panel(path))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#12263a")
            spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01, wspace=0.03)
    target = OUT / "gate_perception_panels.pdf"
    figure.savefig(target, dpi=400)
    plt.close(figure)

    provenance = {
        "figure": "gate_perception_panels.pdf",
        "script": "article_journal/scripts/make_perception_figure.py",
        "operation": "camera-panel crop only; detector overlays preserved unchanged",
        "crop_box": list(PANEL),
        "sources": [str(p.relative_to(ROOT)).replace("\\", "/") for p in PANELS],
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
