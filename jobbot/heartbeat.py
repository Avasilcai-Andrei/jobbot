"""Daily heartbeat: one short message a day with what the bot did in the last 24 h.

If the bot dies, the missing heartbeat is the signal; otherwise silence would look the same as "no matching jobs".
Sent at the first run at or after `notify.heartbeat.time` (local time), once per local day. The day it was last
sent is stored in the DB, so a restart doesn't send it twice.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from .config import Config
from .db import DB
from .guard import KIND_LABELS
from .notify import Notifier

log = logging.getLogger(__name__)

META_KEY = "heartbeat_sent_on"


def due(now_local: datetime, at: str, last_sent_on: str | None) -> bool:
    """True if it is at or past `at` (HH:MM) today and no heartbeat went out today."""
    hh, mm = (int(x) for x in at.split(":"))
    return now_local.time() >= time(hh, mm) and last_sent_on != now_local.date().isoformat()


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def build_message(db: DB, now: datetime) -> str:
    runs = db.runs_since(now - timedelta(hours=24))
    ok = [s for _, _, s in runs if not s.get("crashed")]
    crashed = len(runs) - len(ok)
    total = {k: sum(s.get(k, 0) for s in ok) for k in
             ("fetched", "new", "passed_rules", "scored", "notified", "unscored", "rescored", "held",
              "auto_deferred")}
    since = now - timedelta(hours=24)
    paused_s = db.period_seconds("paused", since, now)
    deferred_s = db.period_seconds("deferred", since, now)
    busy = {k: db.period_seconds(k, since, now) for k in db.period_kinds("busy:", since)}
    busy = {KIND_LABELS.get(k.removeprefix("busy:"), k): s for k, s in busy.items() if s >= 30}
    per_run = round(total["fetched"] / len(ok)) if ok else 0
    if runs:
        ago = int((now - runs[-1][1]).total_seconds() // 60)
        last = f", last {ago} min ago" if ago < 120 else f", last {ago // 60} h ago"
    else:
        last = ""
    failing = [f for f in db.all_feed_states() if f.consecutive_failures > 0]
    lines = [
        f"💓 jobbot daily check (last 24 h): {len(runs)} run{'' if len(runs) == 1 else 's'}{last}"
        + (f", {crashed} crashed" if crashed else "") + ".",
        f"Postings fetched: {per_run:,} per run, {total['new']:,} new.",
        f"Passed rules: {total['passed_rules']} · scored: {total['scored']}"
        + (f" ({total['rescored']} twice)" if total["rescored"] else "")
        + f" · alerted: {total['notified']}" + (f" ({total['unscored']} unscored)" if total["unscored"] else "") + ".",
    ]
    queued = db.count_status("digest")
    if queued:
        lines.append(f"📬 {queued} email job{'' if queued == 1 else 's'} waiting for the daily digest.")
    if busy:
        lines.append("🎮 GPU busy (scoring waited): " + " · ".join(
            f"{label} {_duration(secs)}" for label, secs in sorted(busy.items(), key=lambda kv: -kv[1])) + ".")
    if paused_s or deferred_s or total["held"] or total["auto_deferred"]:
        parts = []
        if paused_s:
            parts.append(f"paused {_duration(paused_s)}")
        if deferred_s:
            parts.append(f"scoring deferred {_duration(deferred_s)}")
        if total["auto_deferred"]:
            parts.append(f"GPU busy (auto-deferred) in {total['auto_deferred']} run{'' if total['auto_deferred'] == 1 else 's'}")
        if total["held"]:
            parts.append(f"{total['held']} job-run{'' if total['held'] == 1 else 's'} waited for scoring")
        lines.append("⏸ " + " · ".join(parts) + ".")
    if failing:
        lines.append("Feeds failing: " + ", ".join(
            f"{f.feed} ({f.consecutive_failures}× {(f.last_error or '')[:60]})" for f in failing[:8])
            + (f" and {len(failing) - 8} more" if len(failing) > 8 else ""))
    else:
        lines.append("Feeds failing: none.")
    if not runs and not paused_s:
        lines.append("⚠️ No runs recorded in the last 24 h.")
    return "\n".join(lines)


async def maybe_send(cfg: Config, db: DB, notifier: Notifier, now: datetime,
                     now_local: datetime | None = None) -> bool:
    """Send today's heartbeat if it is due. Returns True if one was sent."""
    hb = cfg.notify.heartbeat
    now_local = now_local or now.astimezone()
    if not hb.enabled or not due(now_local, hb.time, db.get_meta(META_KEY)):
        return False
    text = build_message(db, now)
    try:
        await notifier.send_text(text)
    except Exception as e:  # try again after the next run
        log.error("Could not send the daily heartbeat: %s", e)
        return False
    db.set_meta(META_KEY, now_local.date().isoformat())
    log.info("Sent the daily heartbeat")
    return True
