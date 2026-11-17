"""One snapshot of everything `jobbot status` (and the control panel) shows. Read-only."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from . import gpu, ollama_ctl
from .config import Config
from .control import Control, describe
from .db import _iso
from .models import short_id


async def gather(cfg: Config, control: Control, now: datetime) -> dict[str, Any]:
    v = control.view(now)
    st = v.state
    out: dict[str, Any] = {
        "mode": st.mode,
        "description": describe(st, now),
        "paused_until": _iso(st.paused_until),
        "defer_kind": st.defer_kind,
        "defer_until": _iso(st.defer_until),
        "bot_alive": v.bot_alive,
        "bot_pid": v.bot.pid,
        "bot_last_seen": _iso(v.bot.tick_at),
        "in_run": v.bot.in_run and v.bot_alive,
        "next_run_at": _iso(v.bot.next_run_at) if v.bot_alive else None,
        "pending": v.pending,
        "last_run_finished": _iso(v.last_run[0]) if v.last_run else None,
        "last_run": v.last_run[1] if v.last_run else None,
        "gpu": None if v.gpu is None else {
            "at": _iso(v.gpu.at), "busy": v.gpu.busy, "reasons": [r.text for r in v.gpu.reasons],
            "kinds": [r.kind for r in v.gpu.reasons], "allowed_at": _iso(v.gpu.allowed_at),
            "quiet_minutes": v.gpu.quiet_minutes, "quiet_window": cfg.control.gpu_quiet_minutes,
            "util_pct": v.gpu.util_pct, "procs": v.gpu.procs},
        "pending_oldest_since": _iso(control.db.oldest_pending()),
        "hold_minutes": cfg.control.defer_max_hold_minutes,
        "scan_interval_minutes": cfg.schedule.interval_minutes,
        "alerts": [{"id": short_id(r["source"], r["job_id"]), "title": r["title"], "company": r["company"],
                    "score": r["score"], "notified_at": r["notified_at"], "mark": r["mark"]}
                   for r in control.db.last_alerts(5)],
        "failing_feeds": [{"feed": f.feed, "failures": f.consecutive_failures, "error": f.last_error}
                          for f in control.db.all_feed_states() if f.consecutive_failures > 0],
        "model": None,
        "ollama_reachable": None,
        "gpu_free_mb": await gpu.free_vram_mb((cfg.control.nvidia_smi,)),
    }
    if cfg.llm.backend == "ollama":
        o = cfg.llm.ollama
        try:
            async with httpx.AsyncClient(base_url=o.url.rstrip("/"), timeout=5.0) as client:
                entry = await ollama_ctl.loaded_model(client, o.model)
            out["ollama_reachable"] = True
            out["model"] = {"name": o.model, "loaded": entry is not None,
                            "size": entry.get("size") if entry else None,
                            "size_vram": entry.get("size_vram") if entry else None,
                            "expires_at": entry.get("expires_at") if entry else None}
        except httpx.HTTPError:
            out["ollama_reachable"] = False
            out["model"] = {"name": o.model, "loaded": False}
    return out


def _ago(t: datetime, now: datetime) -> str:
    secs = max(0, int((now - t).total_seconds()))
    return f"{secs} s ago" if secs < 90 else f"{secs // 60} min ago" if secs < 7200 else f"{secs // 3600} h ago"


def _in(t: datetime, now: datetime) -> str:
    secs = int((t - now).total_seconds())
    return "now" if secs <= 0 else f"{secs} s" if secs < 90 else f"{secs // 60} min"


def gpu_line(info: dict[str, Any], now: datetime) -> str | None:
    """"GPU busy: <reason>, scoring allowed from HH:MM (in N min)" or "GPU quiet for N min"; None if unknown."""
    g = info.get("gpu")
    if g is None:
        return None
    if g["busy"]:
        allowed = datetime.fromisoformat(g["allowed_at"]) if g["allowed_at"] else None
        until = (f"scoring allowed from {allowed.astimezone().strftime('%H:%M')} (in {_in(allowed, now)})"
                 if allowed else "scoring waits until it clears")
        return f"GPU busy: {'; '.join(g['reasons'])}; {until}"
    q = g["quiet_minutes"]
    if q is None:
        return "GPU quiet"
    return f"GPU quiet for {'%g+' % q if q >= g['quiet_window'] else '%g' % q} min"


def pending_lines(info: dict[str, Any], now: datetime) -> list[str]:
    """Why scoring is deferred (if it is), and how long the oldest waiting job has waited against the hold limit."""
    def dt(v: str | None) -> datetime | None:
        return datetime.fromisoformat(v) if v else None

    lines: list[str] = []
    if info["mode"] == "paused":
        lines.append("Why:     manual pause (nothing runs)")
    elif info["mode"] == "deferred":
        lines.append(f"Why:     manual defer: {info['description']}")
    elif info.get("gpu") and info["gpu"]["busy"] and info["pending"]:
        lines.append("Why:     auto-defer: " + gpu_line(info, now))
    n = info["pending"]
    if n:
        since = dt(info["pending_oldest_since"])
        hold = info.get("hold_minutes")
        wait = int((now - since).total_seconds() // 60) if since else 0
        tail = f" of the {hold:g} min hold, then it goes out [unscored]" if hold is not None else " (no hold limit)"
        lines.append(f"Pending: {n} job(s) waiting for a score; the oldest has waited {wait} min{tail}")
    else:
        lines.append("Pending: none waiting for a score")
    return lines


def format_text(info: dict[str, Any], now: datetime) -> str:
    def dt(key: str) -> datetime | None:
        return datetime.fromisoformat(info[key]) if info.get(key) else None

    lines = [f"State:   {info['description']}"]
    seen = dt("bot_last_seen")
    if info["bot_alive"]:
        nxt = dt("next_run_at")
        doing = "scanning now" if info["in_run"] else f"next scan in {_in(nxt, now)}" if nxt else "idle"
        lines.append(f"Bot:     alive (pid {info['bot_pid']}), {doing}")
    else:
        lines.append("Bot:     NOT running" + (f" (last seen {_ago(seen, now)})" if seen else " (never seen)"))
    m = info["model"]
    if m is None:
        lines.append("Model:   no local model (scoring backend is not ollama)")
    elif info["ollama_reachable"] is False:
        lines.append(f"Model:   {m['name']} not loaded (Ollama is not reachable)")
    elif m["loaded"]:
        vram = f", {m['size_vram'] / 1024 ** 3:.1f} GB VRAM" if m.get("size_vram") else ""
        lines.append(f"Model:   {m['name']} loaded{vram}")
    else:
        lines.append(f"Model:   {m['name']} not loaded")
    free = info["gpu_free_mb"]
    lines.append("GPU:     " + (f"{free / 1024:.1f} GB VRAM free" if free is not None else "nvidia-smi unavailable"))
    gl = gpu_line(info, now)
    if gl:
        lines.append(f"GPU use: {gl.removeprefix('GPU ')}")
    lines.extend(pending_lines(info, now))
    lr = info.get("last_run")
    if lr:
        done = dt("last_run_finished")
        lines.append(f"Last run: {_ago(done, now)}: " + ("crashed" if lr.get("crashed") else
                     f"fetched {lr.get('fetched', 0)}, new {lr.get('new', 0)}, passed rules {lr.get('passed_rules', 0)}, "
                     f"scored {lr.get('scored', 0)}, alerted {lr.get('notified', 0)}"))
    return "\n".join(lines)
