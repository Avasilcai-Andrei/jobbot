"""Pause / defer controls shared by the running bot, the CLI and the control panel.

The state lives in the SQLite database (`meta.control`, JSON), so it survives restarts and reboots and every process
sees the same thing. The CLI writes it, the bot reads it at the start of every run, before every scoring call and every
`control.poll_seconds` while idle. A `BEGIN IMMEDIATE` transaction makes the read-modify-write safe across processes.

- paused: nothing runs (no fetching, scoring or sending). Ends at `paused_until` (default 12 h) or on `resume`.
- deferred: fetching and the rules keep running, but nothing is scored; jobs that need a score wait as `pending`.
  Ends when the next run starts (`next_scan`), at `until` (`minutes`) or on `resume` (`resume`).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from .config import Config
from .db import DB, _dt, _iso, utcnow
from .guard import META_GPU, Assessment

log = logging.getLogger(__name__)

META_STATE = "control"
META_BOT = "bot_status"
BOT_ALIVE_SECONDS = 30   # the bot writes its status every `control.poll_seconds`, also while it is scoring


@dataclass
class ControlState:
    paused: bool = False
    paused_until: datetime | None = None
    defer_kind: str | None = None          # next_scan | minutes | resume
    defer_until: datetime | None = None    # only for "minutes"

    @property
    def deferred(self) -> bool:
        return self.defer_kind is not None

    @property
    def mode(self) -> str:
        return "paused" if self.paused else "deferred" if self.deferred else "running"

    def to_json(self) -> str:
        return json.dumps({"paused": self.paused, "paused_until": _iso(self.paused_until),
                           "defer_kind": self.defer_kind, "defer_until": _iso(self.defer_until)})

    @classmethod
    def from_json(cls, raw: str | None) -> "ControlState":
        if not raw:
            return cls()
        try:
            d = json.loads(raw)
            return cls(bool(d.get("paused")), _dt(d.get("paused_until")), d.get("defer_kind"), _dt(d.get("defer_until")))
        except (ValueError, TypeError):
            log.warning("Ignoring unreadable control state %r", raw[:100])
            return cls()


@dataclass
class BotInfo:
    pid: int | None = None
    tick_at: datetime | None = None
    in_run: bool = False
    next_run_at: datetime | None = None
    started_at: datetime | None = None


@dataclass
class View:
    state: ControlState
    bot: BotInfo
    bot_alive: bool
    pending: int
    last_run: tuple[datetime, dict[str, Any]] | None   # (finished_at, stats)
    gpu: Assessment | None   # the bot's latest GPU assessment, None if unknown or stale


def describe(state: ControlState, now: datetime) -> str:
    def left(t: datetime | None) -> str:
        if t is None:
            return ""
        secs = max(0, int((t - now).total_seconds()))
        h, m = divmod(secs // 60, 60)
        return f"{h} h {m:02d} min" if h else f"{m} min" if m else f"{secs} s"
    if state.paused:
        return f"paused, auto-resume in {left(state.paused_until)}" if state.paused_until else "paused until you resume"
    if state.defer_kind == "minutes":
        return f"scoring deferred, {left(state.defer_until)} left"
    if state.defer_kind == "next_scan":
        return "scoring deferred until the next scan"
    if state.defer_kind == "resume":
        return "scoring deferred until you resume"
    return "running"


class Control:
    def __init__(self, db: DB, cfg: Config, clock: Callable[[], datetime] = utcnow):
        self.db = db
        self.cfg = cfg
        self.clock = clock
        # Written into the bot status by the beacon (see cli.cmd_run).
        self.in_run = False
        self.next_run_at: datetime | None = None
        self.started_at = clock()

    # --- state ------------------------------------------------------------

    def read(self) -> ControlState:
        return ControlState.from_json(self.db.get_meta(META_STATE))

    def _update(self, change: Callable[[ControlState], None]) -> tuple[ControlState, ControlState]:
        """Apply `change` atomically; returns (before, after)."""
        conn = self.db.conn
        if conn.in_transaction:
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (META_STATE,)).fetchone()
            before = ControlState.from_json(row["value"] if row else None)
            after = ControlState.from_json(before.to_json())
            change(after)
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (META_STATE, after.to_json()))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return before, after

    def _sync_periods(self, before: ControlState, after: ControlState, at: datetime) -> None:
        for kind, was, now_ in (("paused", before.paused, after.paused), ("deferred", before.deferred, after.deferred)):
            if now_ and not was:
                self.db.open_period(kind, at)
            elif was and not now_:
                self.db.close_period(kind, at)

    def peek(self, now: datetime | None = None) -> ControlState:
        """The state as it is right now (expired timers already off), without writing anything."""
        now = now or self.clock()
        s = self.read()
        if s.paused and s.paused_until is not None and now >= s.paused_until:
            s.paused, s.paused_until = False, None
        if s.defer_kind == "minutes" and s.defer_until is not None and now >= s.defer_until:
            s.defer_kind = s.defer_until = None
        return s

    def refresh(self, now: datetime | None = None) -> ControlState:
        """The current state, after letting a pause or a timed deferral expire."""
        now = now or self.clock()
        state = self.read()
        expired_pause = state.paused and state.paused_until is not None and now >= state.paused_until
        expired_defer = state.defer_kind == "minutes" and state.defer_until is not None and now >= state.defer_until
        if not (expired_pause or expired_defer):
            return state

        def change(s: ControlState) -> None:
            if s.paused and s.paused_until is not None and now >= s.paused_until:
                s.paused, s.paused_until = False, None
            if s.defer_kind == "minutes" and s.defer_until is not None and now >= s.defer_until:
                s.defer_kind = s.defer_until = None
        before, after = self._update(change)
        # A period ends when its timer ran out, not when the bot noticed.
        if before.paused and not after.paused:
            self.db.close_period("paused", before.paused_until or now)
            log.info("Pause ended (auto-resume)")
        if before.deferred and not after.deferred:
            self.db.close_period("deferred", before.defer_until or now)
            log.info("Scoring deferral ended")
        return after

    # --- commands (CLI, panel) ------------------------------------------------

    def pause(self, hours: float | None = None, now: datetime | None = None) -> ControlState:
        """hours None = the configured auto-resume (12 h); 0 = until `resume`."""
        now = now or self.clock()
        hours = self.cfg.control.pause_auto_resume_hours if hours is None else hours

        def change(s: ControlState) -> None:
            s.paused = True
            s.paused_until = now + timedelta(hours=hours) if hours > 0 else None
        before, after = self._update(change)
        self._sync_periods(before, after, now)
        log.info("Paused (%s)", describe(after, now))
        return after

    def resume(self, now: datetime | None = None) -> tuple[ControlState, ControlState]:
        """Ends a pause and any deferral. Returns (before, after)."""
        now = now or self.clock()

        def change(s: ControlState) -> None:
            s.paused, s.paused_until, s.defer_kind, s.defer_until = False, None, None, None
        before, after = self._update(change)
        self._sync_periods(before, after, now)
        if before.mode != "running":
            log.info("Resumed (was %s)", before.mode)
        return before, after

    def defer(self, kind: str, minutes: float | None = None, now: datetime | None = None) -> ControlState:
        now = now or self.clock()
        if kind not in ("next_scan", "minutes", "resume"):
            raise ValueError(f"unknown deferral {kind!r}")

        def change(s: ControlState) -> None:
            s.defer_kind = kind
            s.defer_until = now + timedelta(minutes=minutes) if kind == "minutes" else None
        before, after = self._update(change)
        self._sync_periods(before, after, now)
        log.info("Scoring deferred: %s", describe(after, now))
        return after

    # --- what the bot asks ----------------------------------------------------

    def begin_run(self, now: datetime | None = None) -> ControlState:
        """Called when a run starts: a "next scan" deferral ends here. Returns the state for this run."""
        now = now or self.clock()
        state = self.refresh(now)
        if state.defer_kind == "next_scan":
            def change(s: ControlState) -> None:
                if s.defer_kind == "next_scan":
                    s.defer_kind = s.defer_until = None
            before, after = self._update(change)
            self._sync_periods(before, after, now)
            log.info("Scoring deferral ended (next scan)")
            return after
        return state

    def scoring_block(self, now: datetime | None = None) -> str | None:
        """Why nothing may be scored right now (paused / deferred), or None."""
        now = now or self.clock()
        state = self.refresh(now)
        if state.paused or state.deferred:
            return describe(state, now)
        return None

    # --- bot status (for `status` and the panel) -------------------------------

    def write_bot_status(self, now: datetime | None = None) -> None:
        now = now or self.clock()
        self.db.set_meta(META_BOT, json.dumps({
            "pid": os.getpid(), "tick_at": _iso(now), "in_run": self.in_run,
            "next_run_at": _iso(self.next_run_at), "started_at": _iso(self.started_at)}))

    def bot_info(self) -> BotInfo:
        raw = self.db.get_meta(META_BOT)
        if not raw:
            return BotInfo()
        try:
            d = json.loads(raw)
        except ValueError:
            return BotInfo()
        return BotInfo(d.get("pid"), _dt(d.get("tick_at")), bool(d.get("in_run")), _dt(d.get("next_run_at")),
                       _dt(d.get("started_at")))

    def view(self, now: datetime | None = None) -> View:
        now = now or self.clock()
        bot = self.bot_info()
        alive = bot.tick_at is not None and (now - bot.tick_at).total_seconds() <= BOT_ALIVE_SECONDS
        gpu = Assessment.from_json(self.db.get_meta(META_GPU))
        if gpu is not None and (not alive or (now - gpu.at).total_seconds() > max(90, 3 * self.cfg.control.gpu_sample_seconds)):
            gpu = None
        return View(state=self.peek(now), bot=bot, bot_alive=alive, pending=self.db.count_status("pending"),
                    last_run=self.db.last_run(), gpu=gpu)
