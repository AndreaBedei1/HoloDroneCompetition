# Controllers

A Marine Race Arena controller is any Python object with three methods that
returns four numbers. This page is the contract first and our reference
implementations second — the contract is what the benchmark actually enforces.

---

## 1. What a controller may know

Everything a controller receives is information the vehicle could physically
obtain. This is the *participant information boundary*, and the benchmark
enforces it: in official mode a controller that declares `uses_ground_truth` is
rejected outright, and the observation simply never carries privileged fields.

### At reset

`reset(mission_info)` is called once, before the race, with the mission
assignment:

```python
{
    "participant_id": "bluerov2_01",
    "initial_beacon_id": "B01",
    "total_beacons": 12,
    "laps": 1,
    "command_limits": {
        "surge": [-0.95, 0.95],
        "sway":  [-0.95, 0.95],
        "heave": [-0.95, 0.95],
        "yaw":   [-0.95, 0.95],
    },
}
```

In a fleet run one more static block is present:

```python
"fleet": {
    "participant_order": ["bluerov2_01", "bluerov2_02"],
    "release_index": 1,
    "predecessor_id": "bluerov2_01",
}
```

### At every step

`step(observation)` is called once per control step with exactly these
top-level fields:

```python
{
    "local_time_s": 2.145,
    "sensors": {
        "FrontCamera": ...,     # rendered image (HoloOcean adapter only)
        "DepthSensor": ...,
        "IMUSensor":   ...,
        "DVLSensor":   ...,
    },
    "beacons": [
        {
            "beacon_id":       "B01",
            "bearing_deg":     ...,   # relative to the vehicle
            "elevation_deg":   ...,
            "range_m":         ...,
            "signal_strength": ...,
            "received_at_s":   ...,
        },
    ],
    "comms": {"inbox": [{"from": "bluerov2_01", "payload": ..., "received_at_s": ...}]},
}
```

* `local_time_s` and every reception timestamp are relative to **your** release,
  not to the start of the race.
* `beacons` holds only the packets that physically arrived: each gate beacon
  transmits independently, packets outside range are never delivered, and noise,
  scheduling and dropout are seeded by the run seed, the transmitter, the
  receiver and the transmission index. An empty list is normal.
* `comms` exists only when the inter-vehicle channel is enabled.
* The exact sensor subset follows the participant sensor profile in the track
  file; a contact sensor may also be present as an onboard measurement.

### What you will never receive

True pose, world-frame velocity, exact gate geometry, configured current
vectors, other vehicles' true positions, and every referee decision — including
whether you just passed a gate. **Nobody tells you which gate is next.**
Estimating your own course progression is part of the task.

---

## 2. What a controller must return

A body-frame command, normalized, clamped to `command_limits`:

```python
{"surge": 0.42, "sway": 0.0, "heave": -0.05, "yaw": 0.18}
```

| Axis | Positive direction |
|---|---|
| `surge` | forward |
| `sway` | right |
| `heave` | down |
| `yaw` | turn right (nose to starboard) |

Direct thruster control is also accepted, for vehicles where that is the
natural interface:

```python
{"thrusters": [0.3, 0.3, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0]}
```

---

## 3. Lifecycle

```text
load  →  reset(mission_info)  →  step(observation) × N  →  close()
```

`reset` is called once per race, `step` once per control step at the rate set by
`--dt`, and `close` once at the end — also when the race is aborted, so release
resources there.

Two class attributes declare your controller's status:

| Attribute | Meaning |
|---|---|
| `uses_ground_truth = False` | You read no privileged state. Required for official mode. |
| `debug_only = False` | You are a scored participant, not an inspection tool. |

---

## 4. Writing your own

The smallest controller that respects the contract is
[`docs/examples/minimal_controller.py`](examples/minimal_controller.py):

```python
import math
from typing import Any, Dict

from marine_race_arena.participants.controller_interface import BaseController


class MinimalBeaconController(BaseController):
    debug_only = False
    uses_ground_truth = False

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.expected_beacon = str(mission_info.get("initial_beacon_id", "B01"))
        limits = mission_info.get("command_limits", {})
        self.max_command = float(limits.get("surge", [-0.95, 0.95])[1])

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        packets = [p for p in observation.get("beacons", [])
                   if p["beacon_id"] == self.expected_beacon]
        if not packets:
            return self._clamp({"surge": 0.15, "sway": 0.0, "heave": 0.0, "yaw": 0.15})

        packet = max(packets, key=lambda p: p["received_at_s"])
        bearing = math.radians(packet["bearing_deg"])
        elevation = math.radians(packet["elevation_deg"])
        speed = max(0.15, min(0.6, packet["range_m"] / 8.0))
        return self._clamp({
            "surge": speed * math.cos(bearing),
            "sway": 0.0,
            "heave": 0.6 * math.sin(elevation),
            "yaw": 1.2 * bearing,
        })

    def close(self) -> None:
        pass

    def _clamp(self, command: Dict[str, float]) -> Dict[str, float]:
        limit = self.max_command
        return {axis: max(-limit, min(limit, float(value)))
                for axis, value in command.items()}
```

It homes on one beacon and **never advances past it** — on purpose. Deciding
that a gate is behind you is the hard part, and it is yours.

### The recommended starting point

`LocalCourseTracker` is a reusable, participant-side progression estimator built
from the same onboard signals you already get. It advances through

```text
SEARCH → APPROACH → VISUAL_ALIGN → COMMIT → VERIFY_EXIT → ADVANCE
                                                            │
                                                 next beacon or FINISHED
```

and confirms a passage only on persistent visual alignment, DVL-integrated
forward displacement, a close beacon-range minimum followed by a range
turnaround, persistent disappearance of the aligned gate, and fresh packets
placing the expected beacon behind you. A single dropout or an isolated range
jump cannot advance it.

[`marine_race_arena/controllers/student_template.py`](../marine_race_arena/controllers/student_template.py)
is a complete controller built on it — copy that file and replace the steering.

---

## 5. Running your controller

From a file:

```bash
python -m marine_race_arena.scripts.run_marine_race \
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json \
  --benchmark-task clean_gate \
  --participant-controller docs/examples/minimal_controller.py \
  --controller-class MinimalBeaconController \
  --adapter holoocean --official --headless \
  --seed 0 --dt 0.1 --duration 120 \
  --log-dir results/my_controller
```

From an importable module, either form works:

```bash
  --participant-controller my_package.my_module --controller-class MyController
  --participant-controller my_package.my_module:MyController
```

From a race configuration file, so the choice travels with the experiment:

```jsonc
"controller": {
  "module_or_file": "docs/examples/minimal_controller.py",
  "class": "MinimalBeaconController"
}
```

If your controller loads weights, pass the path with `--controller-model-path`
(or the `MARINE_RACE_RL_MODEL` environment variable). Controllers that do not
accept a model path simply ignore it.

---

## 6. Reference implementations

Four are included, by alias. **None of them is privileged**: they are subject to
exactly the contract above, and they exist so there is always something to race
against.

### `rule_gate_baseline` — Continuous Servo

Keeps correcting visual centring all the way through the aperture. Strongest
where approaches are well conditioned.

### `rule_gate_center_then_commit` — Center-then-Commit

Establishes a stable visual lock first, then holds a committed trajectory
through the gate, so a late image-centroid jump caused by partial near-field
contours cannot deflect the vehicle when the aperture margin is smallest. It
differs from Continuous Servo in exactly this one respect — same observations,
same guidance, same tracker, same confirmation logic.

Neither dominates. Which one wins depends on the circuit, which is the point of
having three of them.

### `leader_follower` — distributed coordination

Wraps either rule controller for fleet racing. Each vehicle broadcasts only

```python
{"local_beacon_index": 4, "local_lap": 1, "local_status": "RUNNING"}
```

over the acoustic channel, with its range-dependent latency and seeded loss. The
predecessor comes from the static release order; a follower yields only while a
fresh predecessor report shows less than the configured gate margin, and missing
or stale reports are fail-open. `LF(1)` — a one-gate margin — is the default.

### Recurrent PPO — a learned example

A frozen `RecurrentPPO` policy over the 27-D onboard observation encoding,
shipped as [`artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip`](../artifacts/paper/ppo/model/)
with its hashes and contracts in `provenance.json`. It is **one reference
implementation, not the reference controller** — it demonstrates that a learned
policy can be integrated and scored through the same boundary. Training code is
not part of this release; the checkpoint is frozen.

Run it (needs `requirements-rl.txt`):

```bash
python -m marine_race_arena.learning.gen2.evaluate_speed_robust \
  --controller recurrent_ppo --track horseshoe_bay \
  --checkpoint artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip \
  --seed 20370909 --out results/ppo_eval
```

### Also available

`student_template` (the tracker-based starting point), `keyboard` / `manual` and
`pygame` manual controllers for inspection and data collection, and `oracle` —
a privileged debug controller that is **rejected in official mode** and must
never be reported as a baseline.

---

## 7. Checklist before you report a result

- [ ] `--adapter holoocean` **without** `--allow-fallback`, so a failed engine is an error instead of a silent downgrade.
- [ ] `--official`, so the participant sensor profile and official timing apply.
- [ ] `uses_ground_truth = False` on your controller.
- [ ] The seed recorded, and the circuit given its nominal duration.
- [ ] The referee's `*_summary.json`, not your own bookkeeping, as the result.
