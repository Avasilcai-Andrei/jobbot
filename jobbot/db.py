"""SQLite persistence: jobs, feed health, LLM result cache."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import Job, short_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    source         TEXT NOT NULL,
    job_id         TEXT NOT NULL,
    feed           TEXT NOT NULL,
    title          TEXT NOT NULL,
    company        TEXT NOT NULL,
    location       TEXT NOT NULL,
    url            TEXT NOT NULL,
    description    TEXT,
    remote         INTEGER,
    extra          TEXT NOT NULL DEFAULT '{}',
    crosspost_key  TEXT NOT NULL,
    posted_at      TEXT,
    first_seen_at  TEXT NOT NULL,
    notified_at    TEXT,
    -- seeded | new | pending (passed the rules, waiting for scoring) | filtered | duplicate | scored | notified | error
    -- | digest (an email job that passed the rules, queued for the daily digest) | digested (it went out in one)
    status         TEXT NOT NULL,
    reason         TEXT,
    priority       TEXT,
    score          INTEGER,
    match_json     TEXT,
    duplicate_of   TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    passed_rules   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, job_id)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_crosspost ON jobs(crosspost_key, first_seen_at);

CREATE TABLE IF NOT EXISTS feeds (
    feed                  TEXT PRIMARY KEY,
    seeded_at             TEXT,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    last_success_at       TEXT,
    last_attempt_at       TEXT,
    last_error            TEXT,
    next_attempt_at       TEXT,
    warned                INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- One row per pipeline run, for the daily heartbeat. Kept for RUNS_KEEP_DAYS.
CREATE TABLE IF NOT EXISTS runs (
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    stats_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Time spent paused / with scoring deferred, for the heartbeat. ended_at is NULL while the period is open.
CREATE TABLE IF NOT EXISTS control_periods (
    kind        TEXT NOT NULL,   -- paused | deferred
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);

-- A good/bad verdict on an alert (`jobbot mark <id> good|bad`), later imported into calibration as yes/no.
CREATE TABLE IF NOT EXISTS marks (
    source     TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    label      TEXT NOT NULL,   -- good | bad
    note       TEXT,
    marked_at  TEXT NOT NULL,
    PRIMARY KEY (source, job_id)
);

-- Mail read by the `email` source, keyed by Message-ID (not by the read flag). Rows are written only after the
-- run has ingested the message's jobs. status: seeded | ok | skipped_unknown | parse_failed
CREATE TABLE IF NOT EXISTS email_messages (
    feed          TEXT NOT NULL,
    message_id    TEXT NOT NULL,
    sender_domain TEXT,
    parser        TEXT,
    received_at   TEXT,
    first_seen_at TEXT NOT NULL,
    jobs_parsed   INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    PRIMARY KEY (feed, message_id)
);
"""
RUNS_KEEP_DAYS = 60

# Statuses that mean "this role already went to (or through) the LLM stage".
CROSSPOST_STATUSES = ("new", "pending", "scored", "notified")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


@dataclass
class FeedState:
    feed: str
    seeded_at: datetime | None = None
    consecutive_failures: int = 0
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    last_error: str | None = None
    next_attempt_at: datetime | None = None
    warned: bool = False


class DB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- jobs -------------------------------------------------------------

    def job_exists(self, key: tuple[str, str]) -> bool:
        cur = self.conn.execute("SELECT 1 FROM jobs WHERE source=? AND job_id=?", key)
        return cur.fetchone() is not None

    def insert_job(self, job: Job, *, status: str, first_seen_at: datetime, crosspost_key: str) -> bool:
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO jobs (source, job_id, feed, title, company, location, url,
                   description, remote, extra, crosspost_key, posted_at, first_seen_at, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job.source, job.job_id, job.feed, job.title, job.company, job.location, job.url,
             job.description, None if job.remote is None else int(job.remote),
             json.dumps(job.extra, ensure_ascii=False), crosspost_key,
             _iso(job.posted_at), _iso(first_seen_at), status),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def update_job(self, key: tuple[str, str], **fields: Any) -> None:
        for k, v in list(fields.items()):
            if isinstance(v, datetime):
                fields[k] = _iso(v)
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE source=? AND job_id=?", (*fields.values(), *key))
        self.conn.commit()

    def jobs_with_status(self, status: str) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY first_seen_at", (status,)
        ).fetchall()
        return [row_to_job(r) for r in rows]

    def count_status(self, status: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status,)).fetchone()[0]

    def attempts(self, key: tuple[str, str]) -> int:
        row = self.conn.execute("SELECT attempts FROM jobs WHERE source=? AND job_id=?", key).fetchone()
        return row["attempts"] if row else 0

    def find_crosspost(self, crosspost_key: str, exclude: tuple[str, str], since: datetime) -> sqlite3.Row | None:
        """An earlier posting of the same role that already reached the LLM stage."""
        return self.conn.execute(
            f"""SELECT source, job_id, feed, status FROM jobs
                WHERE crosspost_key=? AND NOT (source=? AND job_id=?) AND first_seen_at>=?
                  AND passed_rules=1 AND status IN ({",".join("?" * len(CROSSPOST_STATUSES))})
                ORDER BY first_seen_at LIMIT 1""",
            (crosspost_key, *exclude, _iso(since), *CROSSPOST_STATUSES),
        ).fetchone()

    def find_ats_copy(self, crosspost_key: str, since: datetime) -> sqlite3.Row | None:
        """A copy of the same role that came from an ATS feed (not email) and was really looked at: it passed the
        rules, or its description (which the email never has) ruled it out. Copies dropped for their title or
        location don't count: that may just be the same role in another country."""
        return self.conn.execute(
            """SELECT source, job_id, feed, status, reason FROM jobs
               WHERE crosspost_key=? AND source != 'email' AND first_seen_at>=? AND status != 'seeded'
                 AND (passed_rules=1 OR (status='filtered' AND reason NOT LIKE 'title %' AND reason NOT LIKE 'location %'))
               ORDER BY first_seen_at LIMIT 1""", (crosspost_key, _iso(since))).fetchone()

    def find_email_crosspost(self, crosspost_key: str, exclude: tuple[str, str], since: datetime) -> sqlite3.Row | None:
        """The same role already queued for (or sent in) a digest from another email."""
        return self.conn.execute(
            """SELECT source, job_id, feed, status FROM jobs
               WHERE crosspost_key=? AND source='email' AND NOT (source=? AND job_id=?) AND first_seen_at>=?
                 AND status IN ('digest', 'digested') ORDER BY first_seen_at LIMIT 1""",
            (crosspost_key, *exclude, _iso(since))).fetchone()

    def queued_for_digest(self) -> list[Job]:
        return self.jobs_with_status("digest")

    def recent(self, limit: int = 30, status: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM jobs"
        args: tuple[Any, ...] = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY first_seen_at DESC, posted_at DESC LIMIT ?"
        return self.conn.execute(q, (*args, limit)).fetchall()

    # --- alerts and their marks --------------------------------------------

    def alerts_by_short_id(self, short: str) -> list[sqlite3.Row]:
        """Alerted jobs (notified_at set) whose short id is `short`; normally one."""
        rows = self.conn.execute("SELECT * FROM jobs WHERE notified_at IS NOT NULL").fetchall()
        return [r for r in rows if short_id(r["source"], r["job_id"]) == short]

    def set_mark(self, key: tuple[str, str], label: str, note: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO marks (source, job_id, label, note, marked_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(source, job_id) DO UPDATE SET label=excluded.label, note=excluded.note, "
                "marked_at=excluded.marked_at", (*key, label, note, _iso(utcnow())))

    def alerts_with_marks(self, since: datetime | None = None) -> list[sqlite3.Row]:
        """Every alerted job (oldest first) with its mark columns (NULL when not marked yet)."""
        q = ("SELECT j.*, m.label AS mark, m.note AS mark_note, m.marked_at FROM jobs j "
             "LEFT JOIN marks m ON m.source = j.source AND m.job_id = j.job_id WHERE j.notified_at IS NOT NULL")
        args: tuple[Any, ...] = ()
        if since:
            q += " AND j.notified_at >= ?"
            args = (_iso(since),)
        return self.conn.execute(q + " ORDER BY j.notified_at", args).fetchall()

    def run_starts(self) -> list[datetime]:
        """Start times of the runs that really ran (crashed ones fetched nothing), oldest first."""
        rows = self.conn.execute("SELECT started_at, stats_json FROM runs ORDER BY started_at").fetchall()
        return [_dt(r["started_at"]) for r in rows if not json.loads(r["stats_json"]).get("crashed")]

    def latency_rows(self, since: datetime | None = None) -> Iterator[sqlite3.Row]:
        q = "SELECT source, feed, posted_at, first_seen_at, notified_at FROM jobs WHERE status != 'seeded'"
        args: tuple[Any, ...] = ()
        if since:
            q += " AND first_seen_at >= ?"
            args = (_iso(since),)
        yield from self.conn.execute(q, args)

    # --- runs and small key/value state ----------------------------------

    def record_run(self, started_at: datetime, finished_at: datetime, stats: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO runs (started_at, finished_at, stats_json) VALUES (?, ?, ?)",
                              (_iso(started_at), _iso(finished_at), json.dumps(stats)))
            self.conn.execute("DELETE FROM runs WHERE started_at < ?",
                              (_iso(finished_at - timedelta(days=RUNS_KEEP_DAYS)),))

    def runs_since(self, since: datetime) -> list[tuple[datetime, datetime, dict[str, Any]]]:
        rows = self.conn.execute("SELECT * FROM runs WHERE started_at >= ? ORDER BY started_at", (_iso(since),))
        return [(_dt(r["started_at"]), _dt(r["finished_at"]), json.loads(r["stats_json"])) for r in rows]

    def delete_meta(self, key: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM meta WHERE key=?", (key,))

    def oldest_pending(self) -> datetime | None:
        """When the longest-waiting `pending` job was first seen."""
        r = self.conn.execute("SELECT MIN(first_seen_at) AS t FROM jobs WHERE status='pending'").fetchone()
        return _dt(r["t"])

    def last_alerts(self, n: int = 5) -> list[sqlite3.Row]:
        """The n most recent alerts (newest first) with their mark, if any."""
        return self.conn.execute(
            "SELECT j.*, m.label AS mark FROM jobs j LEFT JOIN marks m ON m.source = j.source AND m.job_id = j.job_id "
            "WHERE j.notified_at IS NOT NULL ORDER BY j.notified_at DESC LIMIT ?", (n,)).fetchall()

    def last_run(self) -> tuple[datetime, dict[str, Any]] | None:
        r = self.conn.execute("SELECT finished_at, stats_json FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        return (_dt(r["finished_at"]), json.loads(r["stats_json"])) if r else None

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # --- paused / deferred periods -----------------------------------------

    def open_period(self, kind: str, at: datetime) -> None:
        with self.conn:
            if not self.conn.execute("SELECT 1 FROM control_periods WHERE kind=? AND ended_at IS NULL", (kind,)).fetchone():
                self.conn.execute("INSERT INTO control_periods (kind, started_at) VALUES (?, ?)", (kind, _iso(at)))

    def close_period(self, kind: str, at: datetime) -> None:
        with self.conn:
            self.conn.execute("UPDATE control_periods SET ended_at=? WHERE kind=? AND ended_at IS NULL", (_iso(at), kind))

    def open_period_kinds(self, prefix: str = "") -> list[str]:
        return [r["kind"] for r in self.conn.execute(
            "SELECT DISTINCT kind FROM control_periods WHERE ended_at IS NULL AND kind LIKE ?", (prefix + "%",))]

    def period_kinds(self, prefix: str, since: datetime) -> list[str]:
        """Kinds with a period that overlaps [since, now] (for the heartbeat's per-reason report)."""
        return [r["kind"] for r in self.conn.execute(
            "SELECT DISTINCT kind FROM control_periods WHERE kind LIKE ? AND (ended_at IS NULL OR ended_at >= ?) "
            "ORDER BY kind", (prefix + "%", _iso(since)))]

    def period_seconds(self, kind: str, since: datetime, until: datetime) -> float:
        """Seconds of `kind` between since and until (an open period counts up to `until`)."""
        total = 0.0
        for r in self.conn.execute("SELECT started_at, ended_at FROM control_periods WHERE kind=?", (kind,)):
            start = max(_dt(r["started_at"]), since)
            end = min(_dt(r["ended_at"]) or until, until)
            total += max(0.0, (end - start).total_seconds())
        return total

    def rescore_counts(self, since: datetime | None = None) -> tuple[int, int, int]:
        """(jobs with an LLM score, of which scored twice near the threshold, of which then notified)."""
        q = ("SELECT match_json, status FROM jobs WHERE match_json IS NOT NULL"
             + (" AND first_seen_at >= ?" if since else ""))
        total = twice = notified = 0
        for row in self.conn.execute(q, (_iso(since),) if since else ()):
            total += 1
            if json.loads(row["match_json"]).get("scores"):
                twice += 1
                notified += row["status"] == "notified"
        return total, twice, notified

    # --- feeds ------------------------------------------------------------

    # --- email source ------------------------------------------------------

    def email_seen_ids(self, feed: str) -> set[str]:
        rows = self.conn.execute("SELECT message_id FROM email_messages WHERE feed=?", (feed,)).fetchall()
        return {r["message_id"] for r in rows}

    def record_email_messages(self, rows: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """INSERT OR IGNORE INTO email_messages
                   (feed, message_id, sender_domain, parser, received_at, first_seen_at, jobs_parsed, status)
               VALUES (:feed, :message_id, :sender_domain, :parser, :received_at, :first_seen_at,
                       :jobs_parsed, :status)""",
            rows,
        )
        self.conn.commit()

    def email_unknown_senders(self, since: datetime | None = None) -> list[sqlite3.Row]:
        """Sender domains with no parser: domain, messages, last_seen (for `report`)."""
        return self.conn.execute(
            """SELECT sender_domain, COUNT(*) AS messages, MAX(first_seen_at) AS last_seen
               FROM email_messages WHERE status='skipped_unknown' AND (? IS NULL OR first_seen_at >= ?)
               GROUP BY sender_domain ORDER BY messages DESC""",
            (_iso(since), _iso(since)),
        ).fetchall()

    def feed_state(self, feed: str) -> FeedState:
        row = self.conn.execute("SELECT * FROM feeds WHERE feed=?", (feed,)).fetchone()
        if row is None:
            return FeedState(feed=feed)
        return FeedState(
            feed=feed,
            seeded_at=_dt(row["seeded_at"]),
            consecutive_failures=row["consecutive_failures"],
            last_success_at=_dt(row["last_success_at"]),
            last_attempt_at=_dt(row["last_attempt_at"]),
            last_error=row["last_error"],
            next_attempt_at=_dt(row["next_attempt_at"]),
            warned=bool(row["warned"]),
        )

    def save_feed_state(self, s: FeedState) -> None:
        self.conn.execute(
            """INSERT INTO feeds (feed, seeded_at, consecutive_failures, last_success_at,
                   last_attempt_at, last_error, next_attempt_at, warned)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(feed) DO UPDATE SET seeded_at=excluded.seeded_at,
                   consecutive_failures=excluded.consecutive_failures,
                   last_success_at=excluded.last_success_at, last_attempt_at=excluded.last_attempt_at,
                   last_error=excluded.last_error, next_attempt_at=excluded.next_attempt_at,
                   warned=excluded.warned""",
            (s.feed, _iso(s.seeded_at), s.consecutive_failures, _iso(s.last_success_at),
             _iso(s.last_attempt_at), s.last_error, _iso(s.next_attempt_at), int(s.warned)),
        )
        self.conn.commit()

    def all_feed_states(self) -> list[FeedState]:
        return [self.feed_state(r["feed"]) for r in self.conn.execute("SELECT feed FROM feeds ORDER BY feed")]

    # --- LLM cache --------------------------------------------------------

    def llm_cache_get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT result_json FROM llm_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def llm_cache_put(self, cache_key: str, model: str, result: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, model, result_json, created_at) VALUES (?,?,?,?)",
            (cache_key, model, json.dumps(result, ensure_ascii=False), _iso(utcnow())),
        )
        self.conn.commit()


def row_to_job(r: sqlite3.Row) -> Job:
    return Job(
        source=r["source"],
        job_id=r["job_id"],
        feed=r["feed"],
        title=r["title"],
        company=r["company"],
        location=r["location"],
        url=r["url"],
        description=r["description"],
        posted_at=_dt(r["posted_at"]),
        remote=None if r["remote"] is None else bool(r["remote"]),
        extra=json.loads(r["extra"] or "{}"),
    )


def window_start(days: int) -> datetime:
    return utcnow() - timedelta(days=days)
