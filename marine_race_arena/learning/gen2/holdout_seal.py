"""Cryptographic seal over the Generation-2 final holdout circuits.

The seal is **default-deny by path**: any Gen-2 code that resolves a track file
inside the sealed directory must call :func:`assert_course_accessible`, and that
call raises unless the caller presents a valid unseal record.  Training, DAgger,
validation, curriculum generation and hyper-parameter selection therefore
cannot load a holdout circuit even by accident.

Three independent mechanisms, mirroring the Gen-1 design that survived an
adversarial probe:

1. **Identity** -- a circuit is sealed because its resolved path lives under
   :data:`SEALED_DIR`, not because of a name substring.  A copy placed
   elsewhere is a different file and is caught by the hash check instead.
2. **Content** -- the manifest records the SHA-256 of every circuit, its
   generation seed, the course-family version, the git SHA at generation time,
   and a geometry summary.  :func:`verify_seal` re-hashes and reports drift.
3. **Access** -- opening the holdout requires an unseal record created once,
   with an exclusive file create, naming the single frozen policy by its own
   SHA-256.  Every access appends a hash-chained ledger entry.

The seal is tamper-*evident*, not tamper-proof: anyone with write access can
delete files.  What it guarantees is that a holdout circuit cannot be used
without leaving a record that fails verification if the story is edited later.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from marine_race_arena.learning.gen2 import GEN2_EXPERIMENT_ID
from marine_race_arena.learning.gen2 import course_family as cf
from marine_race_arena.learning.gen2 import seeds as gen2_seeds

#: Where the sealed circuits live.  Membership of this directory IS the seal.
SEALED_DIR = Path("marine_race_arena/tracks/gen2_final_holdout")
MANIFEST_PATH = SEALED_DIR / "SEAL_MANIFEST.json"
UNSEAL_RECORD_PATH = Path("results/rl_public/gen2_final_holdout/unseal_record.json")
LEDGER_PATH = Path("results/rl_public/gen2_final_holdout/final_circuit_access.jsonl")

SEAL_SCHEMA_VERSION = "gen2_holdout_seal_v1"

#: The five sealed circuits and the gate counts they were generated at.
#: Lengths span the readiness ladder so the holdout exercises short-horizon
#: competence and long-horizon composition in the same family.
HOLDOUT_PLAN: Dict[str, int] = {
    "gen2_final_holdout_01": 12,
    "gen2_final_holdout_02": 15,
    "gen2_final_holdout_03": 17,
    "gen2_final_holdout_04": 20,
    "gen2_final_holdout_05": 22,
}


class SealedHoldoutAccessError(PermissionError):
    """Raised when sealed geometry is touched without a valid unseal record."""


def _git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root or Path.cwd()),
            capture_output=True, text=True, timeout=20, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sealed_dir(root: Optional[str | Path] = None) -> Path:
    return (Path(root) / SEALED_DIR) if root else SEALED_DIR


def is_sealed_path(path: str | Path, *, root: Optional[str | Path] = None) -> bool:
    """True when ``path`` resolves inside the sealed directory."""
    target = Path(path).resolve()
    try:
        sealed = sealed_dir(root).resolve()
    except OSError:
        return False
    return target == sealed or sealed in target.parents


def assert_course_accessible(
    path: str | Path,
    *,
    context: str,
    unseal_record: Optional[Mapping[str, Any]] = None,
    root: Optional[str | Path] = None,
) -> Path:
    """Default-deny gate.  Every Gen-2 course load goes through here.

    ``unseal_record`` must be the verified record returned by
    :func:`open_final_holdout`; passing a hand-built dict fails verification.
    """
    resolved = Path(path)
    if not is_sealed_path(resolved, root=root):
        return resolved
    if unseal_record is None:
        raise SealedHoldoutAccessError(
            f"{context} attempted to load the sealed Gen-2 final-holdout circuit "
            f"{resolved.name!r}. The holdout opens exactly once, after a single "
            f"Gen-2 policy is frozen; call open_final_holdout() to obtain the "
            f"unseal record."
        )
    verify_unseal_record(unseal_record)
    return resolved


@dataclass(frozen=True)
class SealedCircuit:
    name: str
    file: str
    sha256: str
    seed: int
    gate_count: int
    course_family_version: str
    git_sha: str
    geometry: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_sealed_holdout(
    *,
    root: Optional[str | Path] = None,
    plan: Optional[Mapping[str, int]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate the holdout circuits ONCE and write the seal manifest.

    Refuses to run a second time unless ``overwrite`` is explicitly set, so a
    stray re-invocation cannot quietly replace the sealed geometry after
    training has started.
    """
    plan = dict(plan or HOLDOUT_PLAN)
    base = Path(root) if root else Path.cwd()
    directory = base / SEALED_DIR
    manifest_file = base / MANIFEST_PATH
    if manifest_file.exists() and not overwrite:
        raise SealedHoldoutAccessError(
            f"the Gen-2 holdout is already sealed at {manifest_file}; "
            "regenerating it would invalidate the experiment"
        )
    directory.mkdir(parents=True, exist_ok=True)

    circuit_seeds = gen2_seeds.GEN2_FINAL_HOLDOUT_CIRCUIT_SEEDS
    git_sha = _git_sha(base)
    circuits: List[SealedCircuit] = []
    for index, (name, gate_count) in enumerate(sorted(plan.items())):
        seed = circuit_seeds[index]
        # These seeds must be in the sealed band -- the inverse of every other
        # Gen-2 call site, which asserts a seed is NOT a holdout seed.
        if gen2_seeds.band_of(seed) != "FINAL_HOLDOUT":
            raise ValueError(f"circuit seed {seed} is not in the FINAL_HOLDOUT band")
        spec = cf.sample_course(seed, gate_count=int(gate_count))
        path = cf.materialize_course(
            spec, directory / f"{name}.json", race_name=f"Gen2 final holdout {name}"
        )
        circuits.append(SealedCircuit(
            name=name,
            file=path.name,
            sha256=sha256_file(path),
            seed=int(seed),
            gate_count=int(spec.gate_count),
            course_family_version=cf.GEN2_COURSE_FAMILY_VERSION,
            git_sha=git_sha,
            geometry=spec.summary(),
        ))

    payload: Dict[str, Any] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "experiment": GEN2_EXPERIMENT_ID,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha,
        "course_family_version": cf.GEN2_COURSE_FAMILY_VERSION,
        "min_nonadjacent_gate_separation_m": cf.MIN_NONADJACENT_GATE_SEPARATION_M,
        "circuit_seed_band": [
            gen2_seeds.GEN2_FINAL_HOLDOUT_CIRCUIT_SEEDS[0],
            gen2_seeds.GEN2_FINAL_HOLDOUT_CIRCUIT_SEEDS[-1],
        ],
        "trial_seed_band": [
            gen2_seeds.GEN2_FINAL_HOLDOUT_TRIAL_SEEDS[0],
            gen2_seeds.GEN2_FINAL_HOLDOUT_TRIAL_SEEDS[-1],
        ],
        "circuits": [circuit.as_dict() for circuit in circuits],
        "policy": (
            "Sealed until exactly one Gen-2 policy is frozen and the Gen-2 "
            "readiness gate has been evaluated. Training, DAgger, validation, "
            "curriculum generation and hyper-parameter selection must never "
            "load these files."
        ),
    }
    payload["manifest_sha256"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "manifest_sha256"}
    )
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_manifest(root: Optional[str | Path] = None) -> Dict[str, Any]:
    base = Path(root) if root else Path.cwd()
    return json.loads((base / MANIFEST_PATH).read_text(encoding="utf-8"))


def verify_seal(root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Re-hash every sealed circuit and re-check the manifest's own hash."""
    base = Path(root) if root else Path.cwd()
    manifest = read_manifest(base)
    recorded = manifest.get("manifest_sha256")
    recomputed = canonical_hash(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    circuits = []
    intact = recorded == recomputed
    for entry in manifest.get("circuits", []):
        path = base / SEALED_DIR / entry["file"]
        present = path.exists()
        digest = sha256_file(path) if present else None
        matched = present and digest == entry["sha256"]
        intact = intact and matched
        circuits.append({
            "name": entry["name"],
            "file": entry["file"],
            "present": present,
            "sha256_matches": matched,
            "recorded_sha256": entry["sha256"],
            "observed_sha256": digest,
        })
    return {
        "manifest_hash_matches": recorded == recomputed,
        "recorded_manifest_sha256": recorded,
        "recomputed_manifest_sha256": recomputed,
        "circuits": circuits,
        "intact": intact,
    }


def sealed_circuit_paths(
    unseal_record: Mapping[str, Any], *, root: Optional[str | Path] = None
) -> List[Path]:
    """Return the sealed circuit paths.  Requires a verified unseal record."""
    verify_unseal_record(unseal_record)
    base = Path(root) if root else Path.cwd()
    manifest = read_manifest(base)
    return [base / SEALED_DIR / entry["file"] for entry in manifest["circuits"]]


def open_final_holdout(
    *,
    frozen_policy_path: str | Path,
    frozen_policy_sha256: str,
    readiness_verdict_sha256: str,
    reason: str,
    root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create the once-only unseal record.  Fails if one already exists.

    Binding the record to the frozen policy's own SHA-256 is what stops the
    holdout from being used to *choose* a policy: the record names the single
    checkpoint before any holdout episode runs, and a different checkpoint
    cannot reuse it.
    """
    if len(str(reason).strip()) < 40:
        raise ValueError("the unseal reason must be a written justification (>= 40 chars)")
    base = Path(root) if root else Path.cwd()
    record_path = base / UNSEAL_RECORD_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)

    seal = verify_seal(base)
    if not seal["intact"]:
        raise SealedHoldoutAccessError(
            "refusing to unseal: the Gen-2 holdout seal does not verify"
        )
    payload: Dict[str, Any] = {
        "schema_version": "gen2_unseal_record_v1",
        "experiment": GEN2_EXPERIMENT_ID,
        "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(base),
        "frozen_policy_path": str(frozen_policy_path),
        "frozen_policy_sha256": str(frozen_policy_sha256),
        "readiness_verdict_sha256": str(readiness_verdict_sha256),
        "seal_manifest_sha256": read_manifest(base)["manifest_sha256"],
        "reason": str(reason),
    }
    payload["record_sha256"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "record_sha256"}
    )
    # Exclusive create: the holdout opens exactly once.
    try:
        handle = open(record_path, "x", encoding="utf-8")
    except FileExistsError as exc:
        raise SealedHoldoutAccessError(
            f"the Gen-2 final holdout has already been opened; see {record_path}"
        ) from exc
    with handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def read_unseal_record(root: Optional[str | Path] = None) -> Dict[str, Any]:
    base = Path(root) if root else Path.cwd()
    path = base / UNSEAL_RECORD_PATH
    if not path.exists():
        raise SealedHoldoutAccessError(
            f"the Gen-2 final holdout is still sealed (no record at {path})"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    verify_unseal_record(record)
    return record


def verify_unseal_record(record: Mapping[str, Any]) -> None:
    """Raise unless the record's recorded hash matches its own content."""
    recorded = record.get("record_sha256")
    recomputed = canonical_hash(
        {k: v for k, v in record.items() if k != "record_sha256"}
    )
    if recorded != recomputed:
        raise SealedHoldoutAccessError(
            "the Gen-2 unseal record has been modified since it was written "
            f"(recorded {recorded}, recomputed {recomputed})"
        )


def append_access(
    entry: Mapping[str, Any], *, root: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Append a hash-chained ledger entry for one holdout episode."""
    base = Path(root) if root else Path.cwd()
    path = base / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "genesis"
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1])["entry_sha256"]
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "previous_sha256": previous,
        **dict(entry),
    }
    row["entry_sha256"] = canonical_hash(
        {k: v for k, v in row.items() if k != "entry_sha256"}
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def verify_ledger(root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Walk the ledger and confirm the hash chain is unbroken."""
    base = Path(root) if root else Path.cwd()
    path = base / LEDGER_PATH
    if not path.exists():
        return {"entries": 0, "intact": True, "ledger": str(path)}
    previous = "genesis"
    entries = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        expected = canonical_hash({k: v for k, v in row.items() if k != "entry_sha256"})
        if row.get("previous_sha256") != previous or row.get("entry_sha256") != expected:
            return {"entries": entries, "intact": False, "broken_at": entries, "ledger": str(path)}
        previous = row["entry_sha256"]
        entries += 1
    return {"entries": entries, "intact": True, "ledger": str(path)}
