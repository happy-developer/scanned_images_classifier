from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def append_run(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def new_timing_ms(start_perf_counter: float) -> float:
    return round((time.perf_counter() - start_perf_counter) * 1000.0, 3)


def now_epoch_ms() -> int:
    return int(time.time() * 1000)
