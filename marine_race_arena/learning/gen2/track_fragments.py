"""Exact contiguous fragments of the three official circuits.

The Gen-2 objective is now practical rather than scientific: make the learned
recurrent controller complete Horseshoe Bay, Vertical Serpent and Mixed
Endurance.  Those circuits were opened by the Gen-1 exploratory holdout and are
no longer pristine, so they may be used for development.

A fragment is a **contiguous window of the real gate sequence, unmodified**:

* gate positions, rotations, passage directions, types and links are copied
  byte-for-byte from the source track;
* no rotation, no translation, no perturbation, no resampling;
* world bounds are inherited unchanged, so an out-of-bounds on a fragment means
  the same thing it means on the full circuit.

Two rules make a window legal:

**Linked gates are never split.**  Vertical Serpent links G08-G09; Mixed
Endurance links G08-G09, G14-G15 and G18-G19.  Cutting between a pair would
leave a dangling ``linked_gate`` reference and change what the gate *is*, so a
window that would split a pair is grown to contain both.

**The inbound state is reconstructed, not invented.**  For a fragment starting
at gate *k > 1* the vehicle is placed just past gate *k-1*, on gate *k-1*'s
centre depth, heading along gate *k-1*'s passage direction -- exactly the pose
it would hold at the instant it finished crossing that gate on the full
circuit.  Spawning it in front of gate *k* instead would delete the very
transition the fragment exists to teach.  A fragment starting at gate 1 keeps
the circuit's own start pose.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

TRACK_DIR = Path("marine_race_arena/tracks")

#: The three development circuits, by short name.
OFFICIAL_TRACKS: Dict[str, str] = {
    "horseshoe_bay": "marine_race_horseshoe_bay.json",
    "vertical_serpent": "marine_race_vertical_serpent.json",
    "mixed_endurance": "marine_race_mixed_endurance.json",
}

#: How far past gate k-1 the vehicle starts, along that gate's passage
#: direction.  Far enough to be clear of the gate structure, close enough that
#: the first thing the policy must do is the real k-1 -> k transition.
INBOUND_CLEARANCE_M = 1.5

#: Seconds allowed per gate in a fragment.  The full circuits allow roughly
#: 40-60 s per gate; this is deliberately generous so a timeout means the
#: policy stalled, not that the budget was tight.
SECONDS_PER_GATE = 60

FRAGMENT_SCHEMA_VERSION = "gen2_track_fragment_v1"


def track_path(track: str, root: Optional[str | Path] = None) -> Path:
    if track not in OFFICIAL_TRACKS:
        raise KeyError(f"unknown official track {track!r}; known: {sorted(OFFICIAL_TRACKS)}")
    base = Path(root) if root else Path.cwd()
    return base / TRACK_DIR / OFFICIAL_TRACKS[track]


def load_track(track: str, root: Optional[str | Path] = None) -> Dict[str, Any]:
    return json.loads(track_path(track, root).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrackFragment:
    """One contiguous, unmodified window of a real circuit."""

    track: str
    start_index: int          # 0-based index into the original gate_sequence
    end_index: int            # inclusive
    gate_ids: Tuple[str, ...]
    inbound_gate_id: Optional[str]   # the gate we exit to enter this fragment
    grown_for_link: bool             # window was extended to keep a pair intact

    @property
    def length(self) -> int:
        return len(self.gate_ids)

    @property
    def name(self) -> str:
        return f"{self.track}_G{self.start_index + 1:02d}-G{self.end_index + 1:02d}"

    @property
    def is_prefix(self) -> bool:
        return self.start_index == 0

    def as_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "length": self.length, "name": self.name}


def _link_partner_index(gates: Sequence[Mapping[str, Any]], index: int) -> Optional[int]:
    linked = gates[index].get("linked_gate")
    if not linked:
        return None
    for other, gate in enumerate(gates):
        if gate["id"] == linked:
            return other
    return None


def _grow_for_links(gates: Sequence[Mapping[str, Any]], start: int, end: int) -> Tuple[int, int, bool]:
    """Extend ``[start, end]`` until no linked pair straddles a boundary."""
    grown = False
    changed = True
    while changed:
        changed = False
        for index in range(start, end + 1):
            partner = _link_partner_index(gates, index)
            if partner is None:
                continue
            if partner < start:
                start, changed, grown = partner, True, True
            elif partner > end:
                end, changed, grown = partner, True, True
    return start, end, grown


def enumerate_fragments(
    track: str,
    lengths: Sequence[int] = (2, 3, 4, 5),
    *,
    root: Optional[str | Path] = None,
    stride: int = 1,
) -> List[TrackFragment]:
    """Every overlapping window of the requested lengths, links kept intact."""
    data = load_track(track, root)
    gates = data["gates"]
    sequence = list(data["track"]["gate_sequence"])
    by_id = {gate["id"]: gate for gate in gates}
    ordered = [by_id[gate_id] for gate_id in sequence]
    total = len(sequence)

    seen: set = set()
    fragments: List[TrackFragment] = []
    for length in sorted({int(v) for v in lengths}):
        if length < 2 or length > total:
            continue
        for start in range(0, total - length + 1, max(1, int(stride))):
            end = start + length - 1
            grown_start, grown_end, grown = _grow_for_links(ordered, start, end)
            key = (grown_start, grown_end)
            if key in seen:
                continue
            seen.add(key)
            fragments.append(TrackFragment(
                track=track,
                start_index=grown_start,
                end_index=grown_end,
                gate_ids=tuple(sequence[grown_start : grown_end + 1]),
                inbound_gate_id=sequence[grown_start - 1] if grown_start > 0 else None,
                grown_for_link=grown,
            ))
    fragments.sort(key=lambda f: (f.length, f.start_index))
    return fragments


def inbound_start_pose(
    data: Mapping[str, Any], fragment: TrackFragment
) -> Tuple[List[float], List[float]]:
    """Return ``(position, rotation_rpy_deg)`` for the fragment's start.

    A prefix fragment keeps the circuit's own start.  Any other fragment starts
    just past its inbound gate, on that gate's centre depth, heading along that
    gate's passage direction -- the pose the vehicle actually holds when it
    finishes crossing that gate on the full circuit.
    """
    if fragment.inbound_gate_id is None:
        start = data["start"]
        return list(start["position"]), list(start.get("rotation_rpy_deg", [0.0, 0.0, 0.0]))

    by_id = {gate["id"]: gate for gate in data["gates"]}
    inbound = by_id[fragment.inbound_gate_id]
    direction = [float(v) for v in inbound["passage_direction"]]
    norm = math.sqrt(sum(v * v for v in direction)) or 1.0
    unit = [v / norm for v in direction]
    centre = [float(v) for v in inbound["position"]]
    position = [
        centre[0] + INBOUND_CLEARANCE_M * unit[0],
        centre[1] + INBOUND_CLEARANCE_M * unit[1],
        centre[2] + INBOUND_CLEARANCE_M * unit[2],
    ]
    rotation = [0.0, 0.0, float(inbound["rotation_rpy_deg"][2])]
    return [round(v, 5) for v in position], rotation


def materialize_fragment(
    fragment: TrackFragment,
    output_path: str | Path,
    *,
    root: Optional[str | Path] = None,
    current_profile_none: bool = True,
) -> Path:
    """Write the fragment as a runnable track, geometry copied verbatim."""
    data = load_track(fragment.track, root)
    by_id = {gate["id"]: gate for gate in data["gates"]}
    kept = [copy.deepcopy(by_id[gate_id]) for gate_id in fragment.gate_ids]

    # Resolve the inbound pose against the FULL circuit: the inbound gate is by
    # definition the one just before the fragment, so it is not among the gates
    # that survive the cut.
    position, rotation = inbound_start_pose(data, fragment)

    data["gates"] = kept
    data["track"]["gate_sequence"] = list(fragment.gate_ids)
    data["finish"] = dict(data.get("finish") or {})
    data["finish"]["gate_id"] = fragment.gate_ids[-1]
    data["race"] = dict(data["race"])
    data["race"]["expected_gates_per_lap"] = len(fragment.gate_ids)
    data["race"]["laps"] = 1
    data["race"]["name"] = f"Gen2 fragment {fragment.name}"
    data["race"]["max_duration_s"] = max(90, len(fragment.gate_ids) * SECONDS_PER_GATE)

    data["start"] = dict(data.get("start") or {})
    data["start"]["position"] = position
    data["start"]["rotation_rpy_deg"] = rotation
    data["participants"] = copy.deepcopy(data["participants"])
    data["participants"][0]["spawn"] = {
        "position": list(position), "rotation_rpy_deg": list(rotation)
    }

    # The official comparison (PPO 0/30 vs rules 30/30) was run current-free,
    # so fragments stay current-free to remain comparable to it.
    #
    # Mixed Endurance declares ``benchmark_task: current_gate``, which requires
    # at least one current.  Dropping the currents without also dropping that
    # task produces a config the loader rejects outright -- the exact trap the
    # Gen-1 sequence generator documents.  Both must move together.
    if current_profile_none:
        data["currents"] = []
        data["benchmark_task"] = {"mode": "clean_gate"}
    data["obstacles"] = list(data.get("obstacles") or [])

    path_points = [position, *(gate["position"] for gate in kept)]
    data["track"]["declared_length_m"] = round(
        sum(math.dist(a, b) for a, b in zip(path_points, path_points[1:])), 3
    )
    data["track"]["length_tolerance_m"] = max(3.0, len(kept) * 1.0)

    # World bounds are inherited unchanged on purpose: an out-of-bounds on a
    # fragment then means exactly what it means on the full circuit.
    data["gen2_fragment"] = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        **fragment.as_dict(),
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def verify_geometry_preserved(
    fragment: TrackFragment, materialized: str | Path, *, root: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Confirm every kept gate is byte-identical to the source circuit.

    This is the guarantee the whole approach rests on: the learner must
    practise the geometry it will actually be evaluated on, so a silent
    rotation or translation would invalidate the campaign.
    """
    source = load_track(fragment.track, root)
    by_id = {gate["id"]: gate for gate in source["gates"]}
    produced = json.loads(Path(materialized).read_text(encoding="utf-8"))
    mismatches: List[Dict[str, Any]] = []
    for gate in produced["gates"]:
        original = by_id.get(gate["id"])
        if original is None:
            mismatches.append({"gate": gate["id"], "reason": "not present in the source circuit"})
            continue
        for field in ("position", "rotation_rpy_deg", "passage_direction", "type", "linked_gate"):
            if original.get(field) != gate.get(field):
                mismatches.append({
                    "gate": gate["id"], "field": field,
                    "source": original.get(field), "fragment": gate.get(field),
                })
    dangling = [
        gate["id"] for gate in produced["gates"]
        if gate.get("linked_gate")
        and gate["linked_gate"] not in {g["id"] for g in produced["gates"]}
    ]
    return {
        "fragment": fragment.name,
        "gates": len(produced["gates"]),
        "identical": not mismatches,
        "mismatches": mismatches,
        "dangling_links": dangling,
        "bounds_inherited": produced["world"]["bounds"] == source["world"]["bounds"],
        "sequence": list(produced["track"]["gate_sequence"]),
    }


def fragment_summary(fragments: Sequence[TrackFragment]) -> Dict[str, Any]:
    by_track: Dict[str, Dict[str, int]] = {}
    for fragment in fragments:
        bucket = by_track.setdefault(fragment.track, {})
        key = str(fragment.length)
        bucket[key] = bucket.get(key, 0) + 1
    return {
        "fragments": len(fragments),
        "by_track_and_length": {
            track: dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
            for track, counts in sorted(by_track.items())
        },
        "grown_for_link": sum(1 for f in fragments if f.grown_for_link),
        "prefix_fragments": sum(1 for f in fragments if f.is_prefix),
        "lengths": sorted({f.length for f in fragments}),
    }


def all_fragments(
    lengths: Sequence[int] = (2, 3, 4, 5),
    *,
    tracks: Sequence[str] = tuple(OFFICIAL_TRACKS),
    root: Optional[str | Path] = None,
) -> List[TrackFragment]:
    out: List[TrackFragment] = []
    for track in tracks:
        out.extend(enumerate_fragments(track, lengths, root=root))
    return out


def fragment_seed(fragment: TrackFragment, trial: int = 0) -> int:
    """A stable, collision-free seed for one fragment trial.

    Derived from the fragment identity so a rerun reproduces the same episode,
    and offset into the Gen-2 TRAIN band so the existing seed guards accept it.
    """
    from marine_race_arena.learning.gen2 import seeds as gen2_seeds

    material = f"{fragment.track}|{fragment.start_index}|{fragment.end_index}|{int(trial)}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    band = gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS
    return band[int.from_bytes(digest[:4], "little") % len(band)]
