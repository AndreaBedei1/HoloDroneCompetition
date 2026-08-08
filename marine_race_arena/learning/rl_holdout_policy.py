"""Sealed final circuits and the procedural TRAIN/VALIDATION/TEST split.

The objective is a controller that generalizes to unseen courses, so the three
official circuits are treated as sealed final holdouts.  They may never reach
training, curriculum construction, replay generation, training probes, checkpoint
ranking or hyperparameter selection -- and neither may rotated, translated or
noise-perturbed copies of them, which would be memorization by another name.

Generalization is measured on procedural seed groups that are disjoint by
construction: each group owns a contiguous, non-overlapping seed band, so a
sampler physically cannot draw a validation or test course while training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

HOLDOUT_POLICY_VERSION = "rl_holdout_policy_v1"

FINAL_CIRCUIT_TRACKS = (
    "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
    "marine_race_arena/tracks/marine_race_vertical_serpent.json",
    "marine_race_arena/tracks/marine_race_mixed_endurance.json",
)
FINAL_CIRCUIT_NAMES = (
    "horseshoe_bay", "vertical_serpent", "mixed_endurance",
)

# Purposes that must never touch a final circuit.
SEALED_PURPOSES = frozenset({
    "train", "training", "curriculum", "replay", "probe",
    "checkpoint_ranking", "hyperparameter", "reward_tuning", "validation",
})
# The only purposes allowed to open one.
FINAL_CIRCUIT_PURPOSES = frozenset({
    "final_holdout", "milestone_holdout", "final_comparison",
})


class SealedCircuitViolation(RuntimeError):
    """Raised when a sealed final circuit is requested for a training purpose."""


def _normalize(track: str | Path) -> str:
    text = str(track).replace("\\", "/").lower()
    return text.rsplit("/", 1)[-1]


def is_final_circuit(track: str | Path) -> bool:
    """Whether a track path is one of the three sealed official circuits."""

    name = _normalize(track)
    if any(name == _normalize(path) for path in FINAL_CIRCUIT_TRACKS):
        return True
    # Also catch copies renamed around the circuit's identity.
    return any(circuit in name for circuit in FINAL_CIRCUIT_NAMES)


def assert_not_final_circuit(track: str | Path, *, purpose: str) -> None:
    """Refuse a sealed circuit for any non-holdout purpose."""

    normalized = str(purpose).strip().lower()
    if not is_final_circuit(track):
        return
    if normalized in FINAL_CIRCUIT_PURPOSES:
        return
    raise SealedCircuitViolation(
        f"{track} is a sealed final circuit and cannot be used for "
        f"purpose={purpose!r}; allowed purposes are "
        f"{sorted(FINAL_CIRCUIT_PURPOSES)}"
    )


def assert_no_final_circuits(tracks: Iterable[str | Path], *, purpose: str) -> None:
    for track in tracks:
        assert_not_final_circuit(track, purpose=purpose)


# --------------------------------------------------------------- seed groups

@dataclass(frozen=True)
class SeedGroup:
    """A contiguous, exclusive seed band for one data role."""

    name: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    def contains(self, seed: int) -> bool:
        return self.start <= int(seed) < self.end

    def seed_at(self, index: int) -> int:
        """Deterministic seed for an index, wrapped inside the band."""

        return self.start + (int(index) % self.size)

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "start": self.start,
                "size": self.size, "end": self.end}


# Bands are far apart so an off-by-a-few bug cannot silently cross them.
SEED_GROUPS: Dict[str, SeedGroup] = {
    "train": SeedGroup("train", 1_000_000, 500_000),
    "validation": SeedGroup("validation", 5_000_000, 50_000),
    "test": SeedGroup("test", 8_000_000, 50_000),
}
DATA_ROLES = tuple(SEED_GROUPS) + ("final_circuits",)


def seed_group(role: str) -> SeedGroup:
    try:
        return SEED_GROUPS[str(role)]
    except KeyError:
        raise ValueError(
            f"unknown data role {role!r}; expected one of {sorted(SEED_GROUPS)}"
        ) from None


def role_of_seed(seed: int) -> Optional[str]:
    for name, group in SEED_GROUPS.items():
        if group.contains(seed):
            return name
    return None


def assert_seed_role(seed: int, *, expected: str) -> None:
    """Refuse a seed that does not belong to the band its role owns."""

    group = seed_group(expected)
    if not group.contains(seed):
        actual = role_of_seed(seed)
        raise ValueError(
            f"seed {seed} belongs to {actual or 'no'} band, not {expected!r} "
            f"[{group.start}, {group.end})"
        )


def groups_are_disjoint() -> bool:
    bands = sorted((g.start, g.end) for g in SEED_GROUPS.values())
    return all(bands[i][1] <= bands[i + 1][0] for i in range(len(bands) - 1))


def evaluation_seeds(role: str, count: int, *, offset: int = 0) -> list:
    """Fixed, reproducible seeds for a non-training role."""

    if role == "train":
        raise ValueError("training seeds are drawn by the sampler, not enumerated")
    group = seed_group(role)
    return [group.seed_at(offset + index) for index in range(int(count))]


# ------------------------------------------------------------ usage ledger

def record_final_circuit_use(
    ledger_path: str | Path,
    *,
    purpose: str,
    circuits: Sequence[str],
    checkpoint: Optional[str] = None,
    transitions: Optional[int] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an auditable record of every sealed-circuit evaluation."""

    from marine_race_arena.learning.provenance import now_utc

    if str(purpose).strip().lower() not in FINAL_CIRCUIT_PURPOSES:
        raise SealedCircuitViolation(
            f"final circuits cannot be used for purpose={purpose!r}"
        )
    record = {
        "schema_version": HOLDOUT_POLICY_VERSION,
        "utc": now_utc(),
        "purpose": str(purpose),
        "circuits": [str(value) for value in circuits],
        "checkpoint": checkpoint,
        "total_environment_transitions": transitions,
        "note": note,
    }
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def policy_manifest() -> Dict[str, Any]:
    return {
        "schema_version": HOLDOUT_POLICY_VERSION,
        "final_circuits": list(FINAL_CIRCUIT_TRACKS),
        "final_circuit_purposes": sorted(FINAL_CIRCUIT_PURPOSES),
        "sealed_purposes": sorted(SEALED_PURPOSES),
        "seed_groups": {name: g.as_dict() for name, g in SEED_GROUPS.items()},
        "seed_groups_disjoint": groups_are_disjoint(),
        "data_roles": list(DATA_ROLES),
    }
