"""Job-alert emails from a dedicated Gmail inbox, read over IMAP (read-only), one parser per sender.

A mailbox is one feed (`email:<name>`). Processed mail is tracked by Message-ID in SQLite, written by
`after_ingest` once the run has stored the jobs. The first run records everything already in the window as seen
(`seeded`) without parsing it. Job pages are never fetched: jobs are scored from the email content alone.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..config import Config
from ..db import DB, utcnow
from ..http import PoliteClient
from ..mailbox import Header, Mailbox, MailboxError
from ..models import Job
from . import register
from .base import Source
from .email_parsers import DEFAULT_SENDERS, PARSERS, ParsedJob, match_sender

log = logging.getLogger(__name__)


@register
class EmailSource(Source):
    kind = "email"
    config_key = "mailboxes"

    def __init__(self, name: str, address: str, password: str, *, host: str = "imap.gmail.com", port: int = 993,
                 folder: str = "INBOX", lookback_days: int = 7, max_messages_per_run: int = 50,
                 senders: dict[str, str] | None = None, imap_factory: Callable[..., Any] | None = None,
                 clock: Callable[[], datetime] = utcnow):
        super().__init__(name=name, company="(email alert)")
        self.address, self.password = address, password
        self.host, self.port, self.folder = host, port, folder
        self.lookback_days, self.max_messages = lookback_days, max_messages_per_run
        self.senders = {**DEFAULT_SENDERS, **(senders or {})}
        self.imap_factory, self.clock = imap_factory, clock
        self.db: DB | None = None
        self._pending: list[dict[str, Any]] = []

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> EmailSource:
        raise NotImplementedError("the email source needs the secrets; use from_config")

    @classmethod
    def from_config(cls, cfg: Config, entry: dict[str, Any]) -> EmailSource | None:
        if not (cfg.secrets.jobmail_address and cfg.secrets.jobmail_app_password):
            log.warning("Email source %r is enabled but JOBMAIL_ADDRESS / JOBMAIL_APP_PASSWORD are not set "
                        "in .env: skipping it", entry.get("name", "jobmail"))
            return None
        keys = ("host", "port", "folder", "lookback_days", "max_messages_per_run", "senders")
        return cls(entry.get("name", "jobmail"), cfg.secrets.jobmail_address, cfg.secrets.jobmail_app_password,
                   **{k: entry[k] for k in keys if k in entry})

    def bind(self, db: DB) -> None:
        self.db = db

    async def fetch(self, http: PoliteClient) -> list[Job]:
        if self.db is None:
            raise RuntimeError("EmailSource.bind(db) was not called")
        self._pending = []
        seeding = self.db.feed_state(self.feed).seeded_at is None
        seen = self.db.email_seen_ids(self.feed)
        jobs, self._pending = await asyncio.to_thread(self._read, seen, seeding)
        return jobs

    def after_ingest(self, now: datetime) -> None:
        if self.db is not None and self._pending:
            self.db.record_email_messages(self._pending)
        self._pending = []

    # --- everything below runs in a worker thread (imaplib is blocking) -----------------------------

    def _read(self, seen: set[str], seeding: bool) -> tuple[list[Job], list[dict[str, Any]]]:
        now = self.clock()
        since = (now - timedelta(days=self.lookback_days)).date()
        jobs: list[Job] = []
        rows: list[dict[str, Any]] = []
        with Mailbox(self.host, self.port, self.address, self.password, self.folder, self.imap_factory) as box:
            fresh: list[Header] = []
            ids: set[str] = set()
            for h in box.headers_since(since):
                h.message_id = h.message_id or _fallback_id(h)
                if h.message_id in seen or h.message_id in ids:
                    continue
                ids.add(h.message_id)
                fresh.append(h)
            if seeding:
                rows = [self._row(h, now, "seeded", None, 0) for h in fresh]
                log.info("First run for %s: marked %d existing messages as seen (no alerts)", self.feed, len(fresh))
                return [], rows
            # Mail from a sender in the table whose parser isn't written yet is left unrecorded, so it is parsed
            # once the parser exists (while it is still inside lookback_days). It does not count towards the cap.
            waiting: dict[str, int] = {}
            ready: list[tuple[Header, tuple[str, str] | None]] = []
            for h in fresh:
                matched = match_sender(h.sender_domain, self.senders)
                if matched is not None and matched[1] not in PARSERS:
                    waiting[matched[1]] = waiting.get(matched[1], 0) + 1
                else:
                    ready.append((h, matched))
            for name, n in waiting.items():
                log.info("%s: %d message(s) waiting for parser %r (not written yet; left unprocessed)",
                         self.feed, n, name)
            if len(ready) > self.max_messages:
                log.info("%s: %d new messages, reading the oldest %d now", self.feed, len(ready), self.max_messages)
                ready = ready[:self.max_messages]
            for h, matched in ready:
                if matched is None:
                    log.info("%s: no parser for sender %s (subject %r): skipped", self.feed,
                             h.sender_domain or h.sender or "?", h.subject[:60])
                    rows.append(self._row(h, now, "skipped_unknown", None, 0))
                    continue
                _, parser_name = matched
                try:
                    parsed = PARSERS[parser_name](box.message(h.uid))
                except MailboxError:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad template must not stop the other mails
                    log.warning("%s: parser %s failed on %r: %s: %s", self.feed, parser_name, h.subject[:60],
                                type(e).__name__, e)
                    rows.append(self._row(h, now, "parse_failed", parser_name, 0))
                    continue
                if not parsed:
                    log.warning("%s: parser %s found no jobs in %r (template changed?)", self.feed,
                                parser_name, h.subject[:60])
                    rows.append(self._row(h, now, "parse_failed", parser_name, 0))
                    continue
                jobs += [self._job(p, parser_name, h) for p in parsed]
                rows.append(self._row(h, now, "ok", parser_name, len(parsed)))
        return jobs, rows

    def _row(self, h: Header, now: datetime, status: str, parser: str | None, parsed: int) -> dict[str, Any]:
        return dict(feed=self.feed, message_id=h.message_id, sender_domain=h.sender_domain, parser=parser,
                    received_at=h.date.astimezone(timezone.utc).isoformat() if h.date else None,
                    first_seen_at=now.astimezone(timezone.utc).isoformat(), jobs_parsed=parsed, status=status)

    def _job(self, p: ParsedJob, parser_name: str, h: Header) -> Job:
        if p.job_id:
            job_id = f"{parser_name}:{p.job_id}"
        else:
            digest = hashlib.sha1("|".join((p.company, p.title, p.location)).lower().encode()).hexdigest()
            job_id = f"{parser_name}:h{digest[:12]}"
        return Job(source="email", job_id=job_id, feed=self.feed, title=p.title, company=p.company,
                   location=p.location, url=p.url, description=p.snippet or "", posted_at=h.date,
                   extra={"_from_email": "1", "_email_platform": parser_name})


def _fallback_id(h: Header) -> str:
    return "nomid:" + hashlib.sha1(f"{h.sender}|{h.date}|{h.subject}".encode()).hexdigest()[:20]
