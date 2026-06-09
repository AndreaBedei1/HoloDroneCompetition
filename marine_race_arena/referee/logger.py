"""JSONL race event logger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RaceLogger:
    def __init__(self, log_dir: str | Path, race_name: str, track_file: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in race_name).strip("_")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.event_path = self.log_dir / f"{safe_name}_{stamp}.jsonl"
        self.summary_path = self.log_dir / f"{safe_name}_{stamp}_summary.json"
        self.track_file = track_file
        self._handle = self.event_path.open("w", encoding="utf-8")

    def log_event(
        self,
        event_type: str,
        time_s: float,
        participant_id: Optional[str] = None,
        **payload: Any,
    ) -> None:
        event: Dict[str, Any] = {
            "event": event_type,
            "time_s": time_s,
        }
        if participant_id is not None:
            event["participant_id"] = participant_id
        event.update(payload)
        self._handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._handle.flush()

    def write_summary(self, summary: Dict[str, Any]) -> None:
        if self.track_file:
            summary.setdefault("track_file", self.track_file)
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    def close(self) -> None:
        self._handle.close()

