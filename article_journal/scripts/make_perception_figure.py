"""Compose the onboard gate-perception figure from the released captures.

Post-processing only. Reads the three HoloOcean diagnostic captures under
``artifacts/paper/perception/captures/`` and writes one PNG into
``article_journal/figures/generated/``. It does not launch the simulator and
does not modify any artifact.

The captures were produced by the interactive perception viewer. This script
keeps only the camera panel crop; detector overlays are preserved unchanged,
while viewer chrome outside the crop is omitted.

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
CAPTURES = ROOT / "artifacts" / "paper" / "perception" / "captures"
OUT = ROOT / "article_journal" / "figures" / "generated"

# Camera panel of the three-view viewer. The upper crop boundary excludes the
# viewer's state badge; the lower boundary is above the localized panel footer.
PANEL = (0, 90, 640, 445)

# Panel order: detection without corners, four-corner lock, multi-candidate lock.
PANELS = [
    CAPTURES / "20260907_191258_mixed_endurance_seed67002__00000_detection_without_corners.png",
    CAPTURES / "20260907_191120_horseshoe_bay_seed67000__00001_corners_B01.png",
    CAPTURES / "20260907_191216_vertical_serpent_seed67001__00035_multi_candidate_lock.png",
]


def build_panel(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").crop(PANEL)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in PANELS if not path.exists()]
    if missing:
        raise SystemExit("missing source capture(s): {}".format(missing))

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 1.70))
    for axis, path in zip(axes, PANELS):
        axis.imshow(build_panel(path))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#12263a")
            spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01, wspace=0.03)
    target = OUT / "gate_perception_panels.png"
    figure.savefig(target, dpi=400)
    plt.close(figure)

    provenance = {
        "figure": "gate_perception_panels.png",
        "script": "article_journal/scripts/make_perception_figure.py",
        "operation": "camera-panel crop only; detector overlays preserved unchanged",
        "crop_box": list(PANEL),
        "sources": [path.relative_to(ROOT).as_posix() for path in PANELS],
        "source_capture_report": "artifacts/paper/perception/visual_smoke_report.json",
        "adapter": "holoocean",
        "seeds": [67002, 67000, 67001],
        "tracks": ["mixed_endurance", "horseshoe_bay", "vertical_serpent"],
        "holoocean_launched_by_this_script": False,
        "artifacts_modified_by_this_script": False,
    }
    with (OUT / "gate_perception_panels.provenance.json").open(
            "w", encoding="utf-8", newline=chr(10)) as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
    print("wrote {}".format(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
