"""Watch what the rover actually perceives, one frame at a time.

    python -m marine_race_arena.tools.obs_debug

Runs Horseshoe Bay and opens one window with five panels: the raw camera, the
annotated camera, the beacon packet, the rover/control state, and a synthesis
panel that says whether vision and the beacon agree.

**Standard mode is onboard only.** Every number on screen is decoded from the
rover's own sensors, its own gate detector, or its own course tracker -- the
same three sources the controller has. The simulator's view of the world is not
read at all unless ``--with-ground-truth`` is passed, and that mode draws its
extra panel in a separate colour with an explicit banner, because a debug tool
that quietly mixes the two is how you end up trusting a number the controller
could never have had.

The tracker's ``diagnostics()`` is used freely: it is onboard-derived state and
is documented in the tracker as debug-only, never fed back into the
observation.

Keys
    SPACE   pause / resume
    n       one frame forward (while paused)
    f       faster        s   slower
    r       toggle real-time throttle
    c       screenshot now
    q, ESC  quit

Outputs land in ``results/debug/<run>/``: a JSONL with every frame's signals, a
PNG per interesting event, and -- with ``--video`` -- an MP4 of the window.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Panel geometry. Deliberately fixed and generous: this is for reading, not
#: for fitting on a laptop screen.
CAM_W, CAM_H = 640, 480
PANEL_H = 330
WIN_W = CAM_W * 2
WIN_H = CAM_H + PANEL_H

#: BGR, because OpenCV.
BG = (24, 24, 28)
FG = (232, 232, 236)
DIM = (150, 150, 158)
OK = (120, 220, 130)
WARN = (70, 190, 245)
BAD = (90, 90, 240)
ACCENT = (240, 190, 90)
VISION = (250, 200, 60)
BEACON = (120, 235, 140)
CENTRE = (200, 200, 205)
TRUTH = (220, 120, 240)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

#: The FrontCamera's horizontal field of view, used to turn a normalized image
#: position into an implied bearing so vision and the beacon can be compared in
#: the same units. This mirrors the detector's own convention.
CAMERA_FOV_DEG = 90.0

EVENT_COLOURS = {
    "gate_acquired": OK,
    "gate_lost": WARN,
    "crossing": ACCENT,
    "target_switch": ACCENT,
    "reacquired": OK,
    "beacon_lost": BAD,
    "disagreement": BAD,
}

#: Vision and the beacon are called inconsistent above this many degrees of
#: implied-bearing mismatch, sustained. One frame of disagreement is normal at
#: close range where the aperture fills the view; a run of them is not.
DISAGREEMENT_DEG = 25.0
DISAGREEMENT_FRAMES = 5


@dataclass
class FrameSignals:
    """One frame of everything the rover knows, plus how it was drawn."""

    step: int
    time_s: float
    # vision, from the rover's own detector on its own camera
    vision_present: bool
    vision_center_x: Optional[float]
    vision_center_y: Optional[float]
    vision_confidence: Optional[float]
    vision_area_fraction: Optional[float]
    vision_width_fraction: Optional[float]
    vision_height_fraction: Optional[float]
    vision_candidates: int
    vision_implied_bearing_deg: Optional[float]
    # beacon, from the received packet for the rover's own expected beacon
    beacon_present: bool
    beacon_id: Optional[str]
    beacon_bearing_deg: Optional[float]
    beacon_elevation_deg: Optional[float]
    beacon_range_m: Optional[float]
    beacon_signal_strength: Optional[float]
    beacon_age_s: Optional[float]
    beacon_bearing_rate_deg_s: Optional[float]
    beacon_elevation_rate_deg_s: Optional[float]
    beacon_range_rate_m_s: Optional[float]
    beacon_rate_valid: bool
    # rover state
    depth_m: Optional[float]
    depth_reference_m: Optional[float]
    depth_error_m: Optional[float]
    dvl_surge: Optional[float]
    dvl_sway: Optional[float]
    dvl_heave: Optional[float]
    dvl_present: bool
    imu_yaw_rate: Optional[float]
    action: Sequence[float]
    # tracker, onboard
    expected_beacon_id: Optional[str]
    tracker_phase: Optional[str]
    tracker_status: Optional[str]
    local_completed: Optional[int]
    target_changed_recently: bool
    time_since_target_change_s: float
    filtered_range_m: Optional[float]
    range_rise_m: Optional[float]
    commit_active: bool
    close_range_confirmed: bool
    # cross-signal
    bearing_disagreement_deg: Optional[float]
    consistent: Optional[bool]
    events: List[str]

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        out = asdict(self)
        out["action"] = [float(v) for v in self.action]
        return out


# --------------------------------------------------------------- reading

def _as_list(value: Any) -> List[float]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(v) for v in arr]


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def implied_bearing_deg(center_x: Optional[float]) -> Optional[float]:
    """Bearing a detection at ``center_x`` implies, in the beacon's convention.

    ``center_x`` is normalized to [-1, 1] across the image. A detection right
    of image centre is to starboard, which is a negative bearing in the
    beacon's frame, hence the sign flip.
    """
    if center_x is None:
        return None
    return -float(center_x) * (CAMERA_FOV_DEG / 2.0)


def read_frame(
    raw: Mapping[str, Any],
    context,
    tracker,
    action: Sequence[float],
    step: int,
) -> FrameSignals:
    """Decode one frame. Reads the rover's sensors and its own tracker only.

    The beacon packet is selected with the encoder's own
    ``_select_beacon_packet``, so the panel shows the exact packet that became
    the observation the controller acted on -- not a second, cleaner reading of
    the same transmitter.
    """
    from marine_race_arena.controllers.vision import (
        select_visual_target_for_beacon,
        vision_targets_from_camera,
    )
    from marine_race_arena.learning.observation_encoder import (
        _depth_m,
        _select_beacon_packet,
    )

    sensors = raw.get("sensors") or {}
    diag = tracker.diagnostics()

    candidates = vision_targets_from_camera(sensors.get("FrontCamera"))
    packet = _select_beacon_packet(
        raw.get("beacons") or [], getattr(context, "expected_beacon_id", None)
    )
    b_bearing = None if packet is None else _finite(packet.get("bearing_deg"))
    b_range = None if packet is None else _finite(packet.get("range_m"))
    target = select_visual_target_for_beacon(candidates, b_bearing, b_range)

    # Positive-down, via the encoder's own conversion. Reading the raw z here
    # and the already-converted reference from the context would put the two on
    # opposite sign conventions and print a depth error of about twice the
    # depth -- a defect in the instrument that reads exactly like a defect in
    # the pipeline.
    depth_m = _depth_m(sensors)
    depth_ref = getattr(context, "depth_reference_m", None)
    dvl = _as_list(sensors.get("DVLSensor"))
    imu = np.asarray(sensors.get("IMUSensor")) if sensors.get("IMUSensor") is not None else None
    yaw_rate = float(imu[1][2]) if imu is not None and imu.shape[0] > 1 else None

    now = _finite(raw.get("local_time_s")) or 0.0
    received = None if packet is None else _finite(packet.get("received_at_s"))
    age = None if received is None else max(0.0, now - received)

    v_bearing = implied_bearing_deg(target.center_x if target else None)
    disagreement = (
        None if (v_bearing is None or b_bearing is None)
        else abs((v_bearing - b_bearing + 180.0) % 360.0 - 180.0)
    )

    return FrameSignals(
        step=step,
        time_s=now,
        vision_present=target is not None,
        vision_center_x=None if target is None else float(target.center_x),
        vision_center_y=None if target is None else float(target.center_y),
        vision_confidence=None if target is None else float(target.confidence),
        vision_area_fraction=None if target is None else float(target.area_fraction),
        vision_width_fraction=None if target is None else float(target.width_fraction),
        vision_height_fraction=None if target is None else float(target.height_fraction),
        vision_candidates=len(candidates),
        vision_implied_bearing_deg=v_bearing,
        beacon_present=packet is not None,
        beacon_id=None if packet is None else str(packet.get("beacon_id")),
        beacon_bearing_deg=b_bearing,
        beacon_elevation_deg=None if packet is None else _finite(packet.get("elevation_deg")),
        beacon_range_m=b_range,
        beacon_signal_strength=(
            None if packet is None else _finite(packet.get("signal_strength"))
        ),
        beacon_age_s=age,
        beacon_bearing_rate_deg_s=getattr(context, "beacon_bearing_rate_deg_s", None),
        beacon_elevation_rate_deg_s=getattr(context, "beacon_elevation_rate_deg_s", None),
        beacon_range_rate_m_s=getattr(context, "beacon_range_rate_m_s", None),
        beacon_rate_valid=bool(getattr(context, "beacon_rate_valid", False)),
        depth_m=depth_m,
        depth_reference_m=_finite(depth_ref),
        depth_error_m=(
            None if (depth_m is None or _finite(depth_ref) is None)
            else depth_m - float(depth_ref)
        ),
        dvl_surge=dvl[0] if len(dvl) > 0 else None,
        dvl_sway=dvl[1] if len(dvl) > 1 else None,
        dvl_heave=dvl[2] if len(dvl) > 2 else None,
        dvl_present=bool(dvl),
        imu_yaw_rate=yaw_rate,
        action=list(action),
        expected_beacon_id=diag.get("expected_beacon_id"),
        tracker_phase=diag.get("phase"),
        tracker_status=diag.get("status"),
        local_completed=diag.get("local_completed"),
        target_changed_recently=bool(getattr(context, "target_changed_recently", False)),
        time_since_target_change_s=float(
            getattr(context, "time_since_target_change_s", 0.0) or 0.0
        ),
        filtered_range_m=_finite(diag.get("filtered_range_m")),
        range_rise_m=_finite(diag.get("range_rise_m")),
        commit_active=bool(diag.get("commit_active")),
        close_range_confirmed=bool(diag.get("close_range_confirmed")),
        bearing_disagreement_deg=disagreement,
        consistent=None if disagreement is None else disagreement <= DISAGREEMENT_DEG,
        events=[],
    )


# --------------------------------------------------------------- drawing

def _text(canvas, s, xy, colour=FG, scale=0.5, thick=1):
    import cv2

    cv2.putText(canvas, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def _label(canvas, s, xy, colour=FG, scale=0.5, thick=1, pad=4):
    """Text on its own dark plate. The seabed is bright; plain text vanishes."""
    import cv2

    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = xy
    cv2.rectangle(canvas, (x - pad, y - h - pad), (x + w + pad, y + base + pad),
                  (18, 18, 22), -1)
    cv2.putText(canvas, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def _fmt(value: Optional[float], spec: str = "%.2f", none: str = "--") -> str:
    return none if value is None else spec % value


def _bar(canvas, x, y, w, h, value, lo, hi, colour):
    """A signed bar with a centre tick, for reading a value at a glance."""
    import cv2

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 66), -1)
    mid = x + w // 2
    cv2.line(canvas, (mid, y), (mid, y + h), (100, 100, 106), 1)
    if value is None:
        return
    frac = max(-1.0, min(1.0, (float(value) - (hi + lo) / 2) / ((hi - lo) / 2 or 1)))
    end = int(mid + frac * (w // 2))
    cv2.rectangle(canvas, (min(mid, end), y), (max(mid, end), y + h), colour, -1)


def camera_bgr(sensors: Mapping[str, Any]) -> np.ndarray:
    """The FrontCamera frame as BGR, or a placeholder when absent."""
    image = sensors.get("FrontCamera")
    if image is None:
        blank = np.full((CAM_H, CAM_W, 3), 40, dtype=np.uint8)
        _text(blank, "NESSUNA IMMAGINE CAMERA", (150, CAM_H // 2), BAD, 0.8, 2)
        return blank
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr.astype(np.uint8))


def draw_annotated(base: np.ndarray, f: FrameSignals) -> np.ndarray:
    """Camera with the detection, the image centre, and the comparison drawn."""
    import cv2

    img = base.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Image centre: crosshair plus a faint full-height line, so an offset gate
    # is obvious without measuring.
    cv2.line(img, (cx, 0), (cx, h), (70, 70, 76), 1)
    cv2.line(img, (0, cy), (w, cy), (70, 70, 76), 1)
    cv2.drawMarker(img, (cx, cy), CENTRE, cv2.MARKER_CROSS, 26, 2)
    _label(img, "+ centro immagine", (12, h - 52), CENTRE, 0.45)

    if f.vision_present and f.vision_center_x is not None:
        gx = int((f.vision_center_x + 1.0) / 2.0 * w)
        gy = int((f.vision_center_y + 1.0) / 2.0 * h)
        bw = int((f.vision_width_fraction or 0.08) * w)
        bh = int((f.vision_height_fraction or 0.08) * h)
        x0, y0 = max(0, gx - bw // 2), max(0, gy - bh // 2)
        x1, y1 = min(w - 1, gx + bw // 2), min(h - 1, gy + bh // 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), VISION, 2)
        cv2.drawMarker(img, (gx, gy), VISION, cv2.MARKER_TILTED_CROSS, 22, 2)
        # The comparison the eye wants: a line from image centre to gate centre.
        cv2.line(img, (cx, cy), (gx, gy), VISION, 1, cv2.LINE_AA)
        dx = f.vision_center_x
        dy = f.vision_center_y
        _label(img, "gate  dx=%+.3f dy=%+.3f" % (dx, dy),
               (x0, max(60, y0 - 26)), VISION, 0.5)
        _label(img, "conf=%.2f  area=%.4f" % (
            f.vision_confidence or 0.0, f.vision_area_fraction or 0.0),
            (x0, max(80, y0 - 6)), VISION, 0.5)
        _label(img, "GATE VISIBILE", (12, 48), OK, 0.7, 2)
    else:
        _label(img, "GATE NON VISIBILE", (12, 48), BAD, 0.7, 2)
    if f.vision_candidates > 1:
        _label(img, "%d candidati nel frame" % f.vision_candidates, (12, 74), DIM, 0.5)

    # Where the beacon says the gate should be, on the same image axis.
    if f.beacon_bearing_deg is not None:
        frac = -f.beacon_bearing_deg / (CAMERA_FOV_DEG / 2.0)
        bx = int((max(-1.0, min(1.0, frac)) + 1.0) / 2.0 * w)
        cv2.line(img, (bx, 0), (bx, h), BEACON, 1)
        _label(img, "| beacon", (12, h - 30), BEACON, 0.45)
    return img


def draw_panels(f: FrameSignals, state: "RunState") -> np.ndarray:
    """The three text panels under the cameras."""
    import cv2

    canvas = np.full((PANEL_H, WIN_W, 3), BG, dtype=np.uint8)
    col = WIN_W // 3
    for i in (1, 2):
        cv2.line(canvas, (col * i, 8), (col * i, PANEL_H - 8), (60, 60, 66), 1)

    # --- panel 3: beacon -------------------------------------------------
    x = 16
    _text(canvas, "BEACON", (x, 26), ACCENT, 0.62, 2)
    if f.beacon_present:
        _text(canvas, "PRESENTE  %s" % (f.beacon_id or "?"), (x, 50), OK, 0.55)
    else:
        _text(canvas, "ASSENTE", (x, 50), BAD, 0.55)
    rows = [
        ("bearing", _fmt(f.beacon_bearing_deg, "%+7.2f deg"), f.beacon_bearing_deg, -60, 60),
        ("elevation", _fmt(f.beacon_elevation_deg, "%+7.2f deg"), f.beacon_elevation_deg, -45, 45),
        ("range", _fmt(f.beacon_range_m, "%7.2f m"), None, 0, 0),
        ("signal", _fmt(f.beacon_signal_strength, "%7.3f"), None, 0, 0),
        ("eta pacchetto", _fmt(f.beacon_age_s, "%7.3f s"), None, 0, 0),
    ]
    y = 76
    for name, shown, bar_value, lo, hi in rows:
        _text(canvas, name, (x, y), DIM, 0.5)
        _text(canvas, shown, (x + 150, y), FG, 0.52)
        if bar_value is not None:
            _bar(canvas, x + 270, y - 10, 130, 12, bar_value, lo, hi, BEACON)
        y += 24
    _text(canvas, "rate %s" % ("validi" if f.beacon_rate_valid else "non validi"),
          (x, y + 4), OK if f.beacon_rate_valid else DIM, 0.5)
    y += 26
    _text(canvas, "d(bear)=%s  d(elev)=%s  d(range)=%s" % (
        _fmt(f.beacon_bearing_rate_deg_s, "%+.2f"),
        _fmt(f.beacon_elevation_rate_deg_s, "%+.2f"),
        _fmt(f.beacon_range_rate_m_s, "%+.2f")), (x, y), DIM, 0.46)

    # --- panel 4: rover and control --------------------------------------
    x = col + 16
    _text(canvas, "ROVER / CONTROLLO", (x, 26), ACCENT, 0.62, 2)
    _text(canvas, "profondita (positiva = giu)", (x, 50), DIM, 0.46)
    _text(canvas, "%s m   rif %s   errore %s" % (
        _fmt(f.depth_m, "%.2f"), _fmt(f.depth_reference_m, "%.2f"),
        _fmt(f.depth_error_m, "%+.2f")), (x, 70), FG, 0.5)
    _text(canvas, "DVL %s" % ("presente" if f.dvl_present else "ASSENTE (valori a zero)"),
          (x, 94), OK if f.dvl_present else WARN, 0.5)
    _text(canvas, "surge %s  sway %s  heave %s" % (
        _fmt(f.dvl_surge, "%+.3f"), _fmt(f.dvl_sway, "%+.3f"),
        _fmt(f.dvl_heave, "%+.3f")), (x, 114), FG, 0.5)
    _text(canvas, "yaw rate IMU  %s" % _fmt(f.imu_yaw_rate, "%+.3f"), (x, 134), FG, 0.5)

    _text(canvas, "azione", (x, 160), DIM, 0.52)
    names = ("surge", "sway", "heave", "yaw")
    ay = 176
    for i, name in enumerate(names):
        value = float(f.action[i]) if i < len(f.action) else 0.0
        _text(canvas, name, (x, ay + 10), DIM, 0.46)
        _bar(canvas, x + 60, ay, 150, 13, value, -1.0, 1.0, WARN)
        _text(canvas, "%+.3f" % value, (x + 220, ay + 11), FG, 0.48)
        ay += 20

    _text(canvas, "target cambiato di recente: %s" % ("SI" if f.target_changed_recently else "no"),
          (x, 268), ACCENT if f.target_changed_recently else DIM, 0.5)
    _text(canvas, "tempo dall'ultimo cambio  %.1f s" % f.time_since_target_change_s,
          (x, 290), FG, 0.5)

    # --- panel 5: synthesis ----------------------------------------------
    x = col * 2 + 16
    _text(canvas, "SINTESI", (x, 26), ACCENT, 0.62, 2)
    _text(canvas, "visione dice   %s" % (
        "gate a %+.1f deg" % f.vision_implied_bearing_deg
        if f.vision_implied_bearing_deg is not None else "nessun gate"),
        (x, 52), VISION if f.vision_present else DIM, 0.52)
    _text(canvas, "beacon dice    %s" % (
        "gate a %+.1f deg, %.1f m" % (f.beacon_bearing_deg, f.beacon_range_m or 0.0)
        if f.beacon_bearing_deg is not None else "nessun pacchetto"),
        (x, 74), BEACON if f.beacon_present else DIM, 0.52)
    _text(canvas, "target onboard %s   fase %s" % (
        f.expected_beacon_id or "?", f.tracker_phase or "?"), (x, 96), FG, 0.52)
    _text(canvas, "gate completati (tracker)  %s" % (
        "--" if f.local_completed is None else f.local_completed), (x, 118), FG, 0.52)

    if f.bearing_disagreement_deg is None:
        _text(canvas, "COERENZA  non valutabile", (x, 150), DIM, 0.56)
    elif f.consistent:
        _text(canvas, "COERENTI  scarto %.1f deg" % f.bearing_disagreement_deg,
              (x, 150), OK, 0.56, 2)
    else:
        _text(canvas, "INCOERENTI  scarto %.1f deg" % f.bearing_disagreement_deg,
              (x, 150), BAD, 0.56, 2)

    _text(canvas, "range filtrato %s   risalita %s   commit %s" % (
        _fmt(f.filtered_range_m, "%.2f"), _fmt(f.range_rise_m, "%.2f"),
        "SI" if f.commit_active else "no"), (x, 176), DIM, 0.46)

    _text(canvas, "eventi recenti", (x, 206), DIM, 0.5)
    ey = 226
    for name, at in state.recent_events[-4:]:
        _text(canvas, "%-16s  passo %d" % (name, at),
              (x, ey), EVENT_COLOURS.get(name, FG), 0.48)
        ey += 20
    if not state.recent_events:
        _text(canvas, "(nessuno)", (x, ey), DIM, 0.48)
    return canvas


def draw_ground_truth_banner(canvas: np.ndarray, truth: Mapping[str, Any]) -> None:
    """Optional comparison strip. Drawn apart and labelled, never mixed in."""
    import cv2

    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 34), (w, h), (48, 24, 52), -1)
    _text(canvas, "CONFRONTO CON GROUND TRUTH -- NON DISPONIBILE AL ROVER",
          (14, h - 12), TRUTH, 0.52, 2)
    _text(canvas, "gate atteso %s   bearing vero %s   range vero %s" % (
        truth.get("gate") or "--",
        _fmt(truth.get("bearing_deg"), "%+.2f deg"),
        _fmt(truth.get("range_m"), "%.2f m")),
        (620, h - 12), TRUTH, 0.52)


def draw_status_bar(canvas: np.ndarray, state: "RunState", f: FrameSignals) -> None:
    import cv2

    cv2.rectangle(canvas, (0, 0), (WIN_W, 24), (38, 38, 44), -1)
    mode = "PAUSA" if state.paused else "PLAY"
    _text(canvas, "%s  passo %d  t=%.1fs  x%.1f%s   [spazio] pausa  [n] avanti  "
                  "[f/s] velocita  [r] tempo reale  [c] screenshot  [q] esci" % (
        mode, f.step, f.time_s, state.speed, "  TEMPO REALE" if state.realtime else ""),
        (12, 17), OK if not state.paused else WARN, 0.46)


# --------------------------------------------------------------- run loop

@dataclass
class RunState:
    paused: bool = False
    step_once: bool = False
    speed: float = 1.0
    realtime: bool = False
    recent_events: List[Tuple[str, int]] = None
    quit: bool = False

    def __post_init__(self):
        if self.recent_events is None:
            self.recent_events = []


def detect_events(f: FrameSignals, previous: Optional[FrameSignals],
                  crossings: int, previous_crossings: int) -> List[str]:
    """Which of the things worth a screenshot happened on this frame."""
    events: List[str] = []
    if previous is None:
        return events
    if f.vision_present and not previous.vision_present:
        events.append("reacquired" if previous.step > 30 else "gate_acquired")
    if previous.vision_present and not f.vision_present:
        events.append("gate_lost")
    if f.expected_beacon_id != previous.expected_beacon_id:
        events.append("target_switch")
    if crossings > previous_crossings:
        events.append("crossing")
    if previous.beacon_present and not f.beacon_present:
        events.append("beacon_lost")
    return events


def run(
    track: str = "horseshoe_bay",
    *,
    seed: int = 67000,
    driver: str = "policy",
    checkpoint: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    max_steps: Optional[int] = None,
    with_ground_truth: bool = False,
    video: bool = False,
    start_paused: bool = False,
    speed: float = 1.0,
) -> Dict[str, Any]:
    import cv2

    from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
    from marine_race_arena.learning.config import ACTION_DIM
    from marine_race_arena.learning.episode import RaceEpisode
    from marine_race_arena.learning.gen2 import track_fragments as tf
    from marine_race_arena.learning.gen2.expert_rollout import (
        action_to_command,
        command_to_action,
    )
    from marine_race_arena.learning.observation_encoder_local_transition import (
        encode_observation_local_transition,
    )
    from marine_race_arena.learning.tracker_context_local_transition import (
        OnboardLocalTransitionContextTracker,
    )

    out = Path(out_dir or (Path("results/debug") / f"{track}_{seed}"))
    out.mkdir(parents=True, exist_ok=True)
    shots = out / "eventi"
    shots.mkdir(exist_ok=True)
    jsonl = (out / "segnali.jsonl").open("w", encoding="utf-8")

    path = tf.track_path(track)
    gate_count = len(tf.load_track(track)["track"]["gate_sequence"])
    steps_cap = int(max_steps or max(2000, gate_count * 900))

    episode = RaceEpisode(
        str(path), seed=int(seed), dt=0.1, adapter="holoocean",
        allow_fallback=False, max_steps=steps_cap, official=True,
        current_profile="none", benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    raw = episode.reset(seed=int(seed))
    config = episode.context.config

    context_source = OnboardLocalTransitionContextTracker(
        total_beacons=max(1, len(config.track.gate_sequence)),
        laps=max(1, int(config.race.laps)),
    )
    context_source.reset(raw)

    controller = None
    expert = None
    if driver == "policy":
        from marine_race_arena.learning.gen2.track_eval import _load_controller

        controller = _load_controller(
            checkpoint or "results/rl/gen2/best/best_completion_policy.zip"
        )
    else:
        from marine_race_arena.controllers.official_baselines import (
            RuleGateCenterThenCommitController,
        )
        from marine_race_arena.scripts.run_marine_race import _mission_info

        expert = RuleGateCenterThenCommitController()
        expert.reset(dict(_mission_info(config, episode.context.participant.id)))

    writer = None
    if video:
        writer = cv2.VideoWriter(
            str(out / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
            (WIN_W, WIN_H + (34 if with_ground_truth else 0)),
        )

    window = "Debug osservazione -- %s (seed %d) -- %s" % (
        track, seed, "SOLO ONBOARD" if not with_ground_truth else "ONBOARD + confronto GT")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, WIN_W, WIN_H + (34 if with_ground_truth else 0))

    state = RunState(paused=bool(start_paused), speed=float(speed))
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    previous: Optional[FrameSignals] = None
    previous_crossings = 0
    disagree_run = 0
    counts: Dict[str, int] = {}
    step = 0

    try:
        while step < steps_cap and not state.quit:
            context = context_source.context(
                raw, dt=0.1, prev_action=previous_action.tolist()
            )
            encoded = encode_observation_local_transition(raw, context)

            if controller is not None:
                action = np.asarray(
                    controller.act(encoded, first_step=(step == 0)), dtype=np.float32
                )
            else:
                action = command_to_action(expert.step(dict(raw)))

            f = read_frame(raw, context, context_source.tracker, action, step)
            crossings = int(episode.referee_progress()["valid_gate_crossings"])

            f.events = detect_events(f, previous, crossings, previous_crossings)
            disagree_run = disagree_run + 1 if f.consistent is False else 0
            if disagree_run == DISAGREEMENT_FRAMES:
                f.events.append("disagreement")
            for name in f.events:
                state.recent_events.append((name, step))
                counts[name] = counts.get(name, 0) + 1

            base = camera_bgr(raw.get("sensors") or {})
            if base.shape[:2] != (CAM_H, CAM_W):
                base = cv2.resize(base, (CAM_W, CAM_H))
            annotated = draw_annotated(base, f)
            top = np.hstack([base, annotated])
            _label(top, "1. CAMERA GREZZA", (12, CAM_H - 12), FG, 0.55, 2)
            _label(top, "2. CAMERA ANNOTATA", (CAM_W + 12, CAM_H - 12), FG, 0.55, 2)
            frame = np.vstack([top, draw_panels(f, state)])

            if with_ground_truth:
                strip = np.full((34, WIN_W, 3), BG, dtype=np.uint8)
                frame = np.vstack([frame, strip])
                draw_ground_truth_banner(frame, _ground_truth(episode))

            draw_status_bar(frame, state, f)
            cv2.imshow(window, frame)

            if f.events:
                cv2.imwrite(str(shots / ("%06d_%s.png" % (step, "_".join(f.events)))), frame)
            if writer is not None:
                writer.write(frame)
            jsonl.write(json.dumps(f.as_dict()) + "\n")

            # Default is as fast as the simulator will go: a HoloOcean step
            # already costs far more than any sane frame delay, so adding one
            # only makes a long circuit longer. 'r' turns on a wall-clock
            # throttle for when the run is too fast to follow by eye.
            wait = max(1, int(100 / max(0.125, state.speed))) if state.realtime else 1
            key = cv2.waitKey(0 if state.paused else wait) & 0xFF
            if key == ord("c"):
                shot = shots / ("manuale_%06d.png" % step)
                cv2.imwrite(str(shot), frame)
                print("screenshot: %s" % shot, flush=True)
            elif not _handle_key(key, state):
                break
            if state.paused and not state.step_once:
                continue
            state.step_once = False

            outcome = episode.step(action_to_command(action))
            raw = outcome.observation
            previous_action = action
            previous = f
            previous_crossings = crossings
            step += 1
            if outcome.terminated or outcome.truncated:
                break
    finally:
        jsonl.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        episode.close()

    summary = {
        "track": track, "seed": seed, "driver": driver, "steps": step,
        "gate_crossings": previous_crossings, "events": counts,
        "onboard_only": not with_ground_truth,
        "output_dir": str(out),
    }
    (out / "riepilogo.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _handle_key(key: int, state: RunState) -> bool:
    if key in (ord("q"), 27):
        state.quit = True
        return False
    if key == ord(" "):
        state.paused = not state.paused
    elif key == ord("n"):
        state.step_once = True
        state.paused = True
    elif key == ord("f"):
        state.speed = min(16.0, state.speed * 2)
        state.realtime = True
    elif key == ord("s"):
        state.speed = max(0.125, state.speed / 2)
        state.realtime = True
    elif key == ord("r"):
        state.realtime = not state.realtime
    return True


def _ground_truth(episode) -> Dict[str, Any]:
    """OPTIONAL comparison only. Never read in standard mode.

    Kept in one function so there is exactly one place where this tool touches
    the simulator, and it is trivially checkable that the standard path does
    not call it.
    """
    from marine_race_arena.learning.gen2.perception_audit import true_geometry

    gate_id = episode.expected_gate_id()
    gate_map = episode.context.arena.gate_map
    if gate_id is None or gate_id not in gate_map:
        return {"gate": gate_id}
    state = episode.context.adapter.get_participant_state(episode.participant_id)
    geo = true_geometry(
        state.position, state.rotation_rpy_deg[2], gate_map[gate_id].center
    )
    return {"gate": gate_id, **geo}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Debug visivo dell'osservazione onboard",
    )
    p.add_argument("--track", default="horseshoe_bay")
    p.add_argument("--seed", type=int, default=67000)
    p.add_argument("--driver", default="policy", choices=("policy", "expert"),
                   help="chi guida: la policy appresa o il controller a regole")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--video", action="store_true", help="salva anche un MP4")
    p.add_argument("--paused", action="store_true", help="parti in pausa")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--with-ground-truth", action="store_true",
                   help="AGGIUNGE una striscia di confronto con la verita del "
                        "simulatore, etichettata. Non e la modalita standard.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        args.track, seed=args.seed, driver=args.driver,
        checkpoint=args.checkpoint, out_dir=args.out, max_steps=args.max_steps,
        with_ground_truth=args.with_ground_truth, video=args.video,
        start_paused=args.paused, speed=args.speed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
