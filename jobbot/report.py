"""`report` command: detection-latency percentiles per source, and feed health."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta

from .db import DB, utcnow, window_start
from .models import short_id

TARGET_MINUTES = 30
OFFLINE_SLACK = timedelta(seconds=60)   # a run starts a few seconds later than "interval after the previous one"


def bot_was_offline(posted_at: str | None, first_seen_at: str, run_starts: list[datetime], interval_minutes: float) -> bool:
    """True if the bot can't have seen a job earlier because it wasn't running.

    A bot that runs every `interval` minutes has a run in every window that long. So if the job was first seen
    at F and there was no run at all in [posted_at, F - interval], the delay is downtime (or a pause), not
    detection delay. When that window is shorter than the interval (plus slack) it can legitimately hold no run, so
    the job counts as detected normally. Jobs first seen before the oldest stored run can't be judged (runs are
    kept 60 days) and also count as normal."""
    if not posted_at or not run_starts:
        return False
    posted, seen = datetime.fromisoformat(posted_at), datetime.fromisoformat(first_seen_at)
    window_end = seen - timedelta(minutes=interval_minutes)
    if seen < run_starts[0] or window_end - posted <= OFFLINE_SLACK:
        return False
    i = bisect_left(run_starts, posted)
    return not (i < len(run_starts) and run_starts[i] <= window_end)


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not sorted_vals:
        raise ValueError("empty")
    pos = (len(sorted_vals) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _minutes(later: str | None, earlier: str | None) -> float | None:
    if not later or not earlier:
        return None
    delta = datetime.fromisoformat(later) - datetime.fromisoformat(earlier)
    return max(0.0, delta.total_seconds() / 60)  # clamp tiny negative clock skew


def _fmt(m: float) -> str:
    return f"{m:.0f}m" if m < 120 else f"{m / 60:.1f}h"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _latency_table(groups: dict[str, list[float]], missing: dict[str, int], offline: dict[str, int] | None = None) -> str:
    offline = offline or {}
    rows = []
    for key in sorted(set(groups) | set(missing) | set(offline)):
        vals = sorted(groups.get(key, []))
        if vals:
            within = sum(v <= TARGET_MINUTES for v in vals) / len(vals) * 100
            rows.append([key, str(len(vals)), str(missing.get(key, 0)), str(offline.get(key, 0)),
                         *(_fmt(percentile(vals, q)) for q in (50, 90, 95)), _fmt(vals[-1]), f"{within:.0f}%"])
        else:
            rows.append([key, "0", str(missing.get(key, 0)), str(offline.get(key, 0)), "-", "-", "-", "-", "-"])
    return _table(["group", "n", "no posted_at", "bot offline", "p50", "p90", "p95", "max", f"<={TARGET_MINUTES}m"], rows)


def build_report(db: DB, days: int | None = None, by: str = "source", interval_minutes: float = 10) -> str:
    since = window_start(days) if days else None
    starts = db.run_starts()
    off_detect: dict[str, int] = defaultdict(int)
    off_notify: dict[str, int] = defaultdict(int)
    detect: dict[str, list[float]] = defaultdict(list)
    notify: dict[str, list[float]] = defaultdict(list)
    no_posted: dict[str, int] = defaultdict(int)
    no_posted_notified: dict[str, int] = defaultdict(int)
    for r in db.latency_rows(since):
        key = r[by]
        offline = bot_was_offline(r["posted_at"], r["first_seen_at"], starts, interval_minutes)
        if (m := _minutes(r["first_seen_at"], r["posted_at"])) is not None:
            if offline:
                off_detect[key] += 1
            else:
                detect[key].append(m)
        else:
            no_posted[key] += 1
        if r["notified_at"]:
            if offline:
                off_notify[key] += 1
            elif (m := _minutes(r["notified_at"], r["posted_at"])) is not None:
                notify[key].append(m)
            else:
                no_posted_notified[key] += 1

    scope = f"last {days} days" if days else "all time"
    out = [
        f"jobbot latency report ({scope}, generated {utcnow():%Y-%m-%d %H:%M} UTC)",
        "Postings present on a feed's first run (seeded) are excluded. 'bot offline' jobs are also left out of the",
        f"percentiles: no run happened between posted_at and first_seen_at minus the {interval_minutes:g}-minute interval,",
        "so the delay was downtime or a pause, not detection.",
        "",
        "Detection latency: posted_at (from the source) -> first_seen_at",
        _latency_table(detect, no_posted, off_detect) if detect or no_posted or off_detect else "  no data yet",
        "",
        "Alert latency: posted_at -> notified_at (notified jobs only)",
        _latency_table(notify, no_posted_notified, off_notify) if notify or no_posted_notified or off_notify
        else "  no alerts yet",
        "",
    ]
    total, twice, twice_notified = db.rescore_counts(since)
    out += [
        "Score stability: scored a second time because the first score was near the threshold",
        f"  {twice} of {total} scored jobs ({twice / total:.0%}); {twice_notified} of those were alerted"
        if total else "  no scored jobs yet",
        "",
        "Feed health",
    ]
    states = db.all_feed_states()
    if states:
        out.append(_table(
            ["feed", "fails in a row", "last success", "last error"],
            [[s.feed, str(s.consecutive_failures),
              f"{s.last_success_at:%m-%d %H:%M}" if s.last_success_at else "never",
              (s.last_error or "")[:60]] for s in states],
        ))
    else:
        out.append("  no runs yet")
    unknown = db.email_unknown_senders(since)
    if unknown:
        out += ["", "Email senders without a parser (send me a sample .eml of one to add it)",
                _table(["sender domain", "messages", "last seen (UTC)"],
                       [[r["sender_domain"] or "?", str(r["messages"]), r["last_seen"][:16].replace("T", " ")]
                        for r in unknown])]
    return "\n".join(out)


def build_marks_report(db: DB, days: int | None = None) -> str:
    """Alerts with their good/bad marks (`jobbot mark`), and the ones not marked yet."""
    rows = db.alerts_with_marks(window_start(days) if days else None)
    if not rows:
        return "No alerts sent" + (f" in the last {days} days." if days else " yet.")
    marked = [r for r in rows if r["mark"]]
    todo = [r for r in rows if not r["mark"]]
    good = sum(r["mark"] == "good" for r in marked)

    def cells(r) -> list[str]:
        score = str(r["score"]) if r["score"] is not None else "-"
        title = f"{r['title']} @ {r['company']}"
        return [short_id(r["source"], r["job_id"]), r["mark"] or "", score, r["notified_at"][:16].replace("T", " "),
                title[:60], r["mark_note"] or ""]

    headers = ["id", "mark", "score", "alerted (UTC)", "job", "note"]
    out = [f"Alerts{f' in the last {days} days' if days else ''}: {len(rows)}; marked {len(marked)} "
           f"({good} good, {len(marked) - good} bad), not marked yet {len(todo)}."]
    if marked:
        out += ["", "Marked:", _left_table(headers, [cells(r) for r in marked])]
    if todo:
        out += ["", "Not marked yet (python -m jobbot mark <id> good|bad [note]):",
                _left_table(headers[:5], [cells(r)[:5] for r in todo])]
    return "\n".join(out)


def _left_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)])
