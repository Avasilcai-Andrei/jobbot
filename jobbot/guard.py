"""Is the GPU busy with something else? Decides whether jobbot may load its model and score.

"Busy" is any of (see `control:` in config.yaml):
- `other_model`: Ollama has a model loaded that is not jobbot's own (another project shares the server);
- `util`: GPU utilization was above `gpu_busy_util` in a sample of the last `gpu_quiet_minutes`, not counting the
  time jobbot itself was scoring, so scoring starts only after the GPU has been quiet for that long;
- `vram`: free VRAM is below `min_free_vram_gb` while jobbot's model is not loaded yet;
- `process`: a running process matches `busy_process_patterns` (name or command line);
- `flag`: an explicit flag, set by `jobbot busy --set` (stored in the database) or by a lock file at
  `busy_lock_path` that other projects create and delete. A lock whose PID is dead, or that is older than
  `busy_max_hours`, is ignored.

While the bot runs, `GpuGuard.run()` samples every `gpu_sample_seconds`, keeps the last 15 minutes in memory, and
publishes the assessment to `meta.gpu_status` for `status` and the panel. jobbot never unloads models it didn't load.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable, Iterator

import httpx

from . import gpu
from .config import Config
from .db import DB, _dt, _iso, utcnow

log = logging.getLogger(__name__)

META_GPU = "gpu_status"
META_FLAG = "busy_flag"
HISTORY = timedelta(minutes=15)
OWN_SCORING_GRACE = timedelta(seconds=5)   # a sample this soon after scoring still measures our own work
KIND_LABELS = {"other_model": "another model loaded", "util": "GPU in use", "vram": "low free VRAM",
               "process": "busy process", "flag": "busy flag"}

Process = tuple[int, str, str]   # pid, name, command line


@dataclass
class Reason:
    kind: str    # a key of KIND_LABELS
    text: str    # shown in status/panel, may change from sample to sample
    key: str     # stable identity, so a change is logged once


@dataclass
class Assessment:
    at: datetime
    reasons: list[Reason] = field(default_factory=list)
    allowed_at: datetime | None = None    # when the util quiet window ends, if that is what holds scoring back
    quiet_minutes: float | None = None    # how long the GPU has been quiet (observed, at most the window)
    util_pct: int | None = None
    free_mb: int | None = None
    min_mb: int = 0
    procs: list[str] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        return bool(self.reasons)

    def describe(self) -> str:
        text = "GPU busy: " + "; ".join(r.text for r in self.reasons)
        return text + (f"; scoring allowed from {self.allowed_at.astimezone().strftime('%H:%M')}"
                       if self.allowed_at else "")

    def to_json(self) -> str:
        return json.dumps({"at": _iso(self.at), "reasons": [vars(r) for r in self.reasons],
                           "allowed_at": _iso(self.allowed_at), "quiet_minutes": self.quiet_minutes,
                           "util_pct": self.util_pct, "free_mb": self.free_mb, "min_mb": self.min_mb,
                           "procs": self.procs})

    @classmethod
    def from_json(cls, raw: str | None) -> "Assessment | None":
        try:
            d = json.loads(raw or "")
            return cls(_dt(d["at"]), [Reason(**r) for r in d["reasons"]], _dt(d.get("allowed_at")),
                       d.get("quiet_minutes"), d.get("util_pct"), d.get("free_mb"), d.get("min_mb", 0),
                       d.get("procs", []))
        except (ValueError, KeyError, TypeError):
            return None


@dataclass
class Sample:
    at: datetime
    util_pct: int
    own: bool   # taken while jobbot was scoring (or just after), so its utilization is ours


# --- explicit busy flag and lock file --------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """Whether a process with this PID exists. (`os.kill(pid, 0)` would terminate it on Windows, so ask psutil.)"""
    import psutil
    return psutil.pid_exists(pid)


def set_flag(db: DB, reason: str, hours: float | None, now: datetime) -> None:
    db.set_meta(META_FLAG, json.dumps({"reason": reason, "set_at": _iso(now),
                                       "until": _iso(now + timedelta(hours=hours)) if hours else None}))


def clear_flag(db: DB) -> bool:
    had = db.get_meta(META_FLAG) is not None
    db.delete_meta(META_FLAG)
    return had


def read_flag(db: DB, now: datetime, max_hours: float) -> tuple[str | None, str | None]:
    """(reason, None) while the flag counts; (None, why) when it exists but is stale; (None, None) when absent."""
    raw = db.get_meta(META_FLAG)
    if not raw:
        return None, None
    try:
        d = json.loads(raw)
        set_at, until = _dt(d["set_at"]), _dt(d.get("until"))
    except (ValueError, KeyError, TypeError):
        return None, "unreadable busy flag"
    if until is not None:
        return (d.get("reason") or "busy flag", None) if now < until else (None, "busy flag expired")
    if now - set_at > timedelta(hours=max_hours):
        return None, f"busy flag older than {max_hours:g} h"
    return d.get("reason") or "busy flag", None


def read_lock(path: Path, now: datetime, max_hours: float,
              alive: Callable[[int], bool] = pid_alive) -> tuple[str | None, str | None]:
    """The lock file other projects create while they use the GPU. Content is optional: JSON like
    {"reason": "...", "pid": 1234, "expires_at": "<ISO time>"}, or plain text (first line = reason; "pid=1234" anywhere
    names the owner). Returns (reason, None) while it counts, (None, why) when it is stale, (None, None) if absent."""
    try:
        stat = path.stat()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    reason, pid, expires = None, None, None
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            reason, pid, expires = d.get("reason"), d.get("pid"), _dt(d.get("expires_at"))
    except ValueError:
        reason = text.strip().splitlines()[0][:120] if text.strip() else None
        m = re.search(r"\bpid\s*[=:]\s*(\d+)", text, re.I)
        pid = int(m.group(1)) if m else None
    reason = str(reason or f"lock file {path.name}")
    created = datetime.fromtimestamp(stat.st_mtime, tz=now.tzinfo)
    if expires is not None and now >= expires:
        return None, f"lock {path.name} expired"
    if now - created > timedelta(hours=max_hours):
        return None, f"lock {path.name} older than {max_hours:g} h"
    if isinstance(pid, int) and not alive(pid):
        return None, f"lock {path.name} owner (PID {pid}) is no longer running"
    return reason, None


# --- default readers ---------------------------------------------------------------------------

def list_processes() -> list[Process]:
    import psutil
    out: list[Process] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            out.append((p.info["pid"], p.info["name"] or "", " ".join(p.info["cmdline"] or [])))
        except (psutil.Error, TypeError):
            continue
    return out


class GpuGuard:
    def __init__(self, cfg: Config, db: DB, *, own_model: str | None = None,
                 clock: Callable[[], datetime] = utcnow,
                 read_gpu: Callable[[], Awaitable[gpu.GpuReading | None]] | None = None,
                 loaded_models: Callable[[], Awaitable[list[str] | None]] | None = None,
                 processes: Callable[[], list[Process]] = list_processes,
                 pid_is_alive: Callable[[int], bool] = pid_alive):
        self.cfg, self.db, self.clock = cfg, db, clock
        self.own_model = own_model
        self.read_gpu = read_gpu or (lambda: gpu.read_gpu((cfg.control.nvidia_smi,)))
        self.loaded_models = loaded_models or self._ollama_models
        self.processes = processes
        self.pid_is_alive = pid_is_alive
        self.samples: deque[Sample] = deque()
        self.first_sample_at: datetime | None = None
        self._scoring = 0
        self._scoring_ended: datetime | None = None
        self._logged: tuple[str, ...] | None = None
        self._stale_logged: set[str] = set()
        self.last: Assessment | None = None
        self.cleared = False   # set when the GPU stops being busy; the bot's idle loop then scores what waited

    # --- own scoring is not "someone else using the GPU" ---

    @contextmanager
    def scoring(self) -> Iterator[None]:
        self._scoring += 1
        try:
            yield
        finally:
            self._scoring -= 1
            self._scoring_ended = self.clock()

    def _own_activity(self, now: datetime) -> bool:
        return self._scoring > 0 or (self._scoring_ended is not None and now - self._scoring_ended < OWN_SCORING_GRACE)

    # --- readers ---

    async def _ollama_models(self) -> list[str] | None:
        """Names of every model loaded in the Ollama server (None if it can't be reached)."""
        if self.cfg.llm.backend != "ollama":
            return None
        try:
            async with httpx.AsyncClient(base_url=self.cfg.llm.ollama.url.rstrip("/"), timeout=5.0) as c:
                resp = await c.get("/api/ps")
                resp.raise_for_status()
                return [m.get("name") or m.get("model") or "?" for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return None

    def _is_own(self, name: str) -> bool:
        return bool(self.own_model) and name in (self.own_model, f"{self.own_model}:latest")

    # --- the decision ---

    async def assess(self, now: datetime | None = None) -> Assessment:
        now = now or self.clock()
        c = self.cfg.control
        a = Assessment(at=now, min_mb=int(c.min_free_vram_gb * 1024))

        flag, stale = read_flag(self.db, now, c.busy_max_hours)
        self._note_stale("flag", stale)
        if flag:
            a.reasons.append(Reason("flag", f"busy flag set: {flag}", f"flag:{flag}"))
        lock, stale = read_lock(self.cfg.resolve(Path(c.busy_lock_path)), now, c.busy_max_hours, self.pid_is_alive)
        self._note_stale("lock", stale)
        if lock:
            a.reasons.append(Reason("flag", f"busy lock file: {lock}", f"lock:{lock}"))
        if not c.auto_defer:
            return a

        reading = await self.read_gpu()
        loaded = await self.loaded_models()
        if reading is not None:
            if self.first_sample_at is None:
                self.first_sample_at = now
            self.samples.append(Sample(now, reading.util_pct, self._own_activity(now)))
            while self.samples and now - self.samples[0].at > HISTORY:
                self.samples.popleft()
            a.util_pct, a.free_mb, a.procs = reading.util_pct, reading.free_mb, reading.procs

        others = sorted(n for n in (loaded or []) if not self._is_own(n))
        if others:
            a.reasons.append(Reason("other_model", f"another Ollama model is loaded ({', '.join(others)})",
                                    "other_model:" + ",".join(others)))

        window = timedelta(minutes=c.gpu_quiet_minutes)
        busy_samples = [s for s in self.samples if not s.own and s.util_pct > c.gpu_busy_util and now - s.at < window]
        if busy_samples:
            last = busy_samples[-1]
            a.allowed_at = last.at + window
            a.reasons.append(Reason("util", f"GPU utilization was {last.util_pct}% at "
                                    f"{last.at.astimezone().strftime('%H:%M')} (limit {c.gpu_busy_util:g}%)", "util"))
        elif self.first_sample_at is not None:
            idle_since = max((s.at for s in self.samples if not s.own and s.util_pct > c.gpu_busy_util),
                             default=self.first_sample_at)
            a.quiet_minutes = round(min((now - idle_since), window).total_seconds() / 60, 1)

        own_loaded = any(self._is_own(n) for n in (loaded or []))
        # While we score, the VRAM drop is our own model loading (it may not be in /api/ps yet).
        if reading is not None and not own_loaded and not self._own_activity(now) and reading.free_mb < a.min_mb:
            a.reasons.append(Reason("vram", f"only {reading.free_mb / 1024:.1f} GB of VRAM is free "
                                    f"(need {c.min_free_vram_gb:g} GB)", "vram"))

        for pattern in c.busy_process_patterns:
            hit = self._match_process(pattern)
            if hit:
                a.reasons.append(Reason("process", f"{hit} is running (matches {pattern!r})", f"process:{pattern}"))
        return a

    def _match_process(self, pattern: str) -> str | None:
        needle, me = pattern.lower(), {os.getpid(), os.getppid()}
        try:
            procs = self.processes()
        except Exception as e:
            log.debug("Could not list processes: %s", e)
            return None
        for pid, name, cmdline in procs:
            if pid not in me and (needle in name.lower() or needle in cmdline.lower()):
                return name or f"PID {pid}"
        return None

    def _note_stale(self, what: str, why: str | None) -> None:
        if why and why not in self._stale_logged:
            self._stale_logged.add(why)
            log.warning("Ignoring a stale busy marker: %s", why)

    async def check(self) -> Assessment:
        """assess() + publish: log a change once, track busy time per reason, store it for status and the panel."""
        a = await self.assess()
        self.publish(a)
        return a

    def publish(self, a: Assessment) -> None:
        if self.last is not None and self.last.busy and not a.busy:
            self.cleared = True
        self.last = a
        sig = tuple(sorted(r.key for r in a.reasons))
        if sig != self._logged:
            if a.busy:
                log.info("%s", a.describe())
            elif self._logged is not None:
                log.info("GPU no longer busy: scoring is allowed")
            self._logged = sig
        want = {f"busy:{r.kind}" for r in a.reasons}
        for kind in self.db.open_period_kinds("busy:"):
            if kind not in want:
                self.db.close_period(kind, a.at)
        for kind in want:
            self.db.open_period(kind, a.at)
        self.db.set_meta(META_GPU, a.to_json())

    async def run(self) -> None:
        """The sampler: runs for the life of the bot."""
        interval = self.cfg.control.gpu_sample_seconds
        # Periods left open by a previous bot process (crash, reboot) end at its last sample.
        previous = Assessment.from_json(self.db.get_meta(META_GPU))
        for kind in self.db.open_period_kinds("busy:"):
            self.db.close_period(kind, previous.at if previous else self.clock())
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("GPU sample failed")
            await asyncio.sleep(interval)
