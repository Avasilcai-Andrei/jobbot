"""Read-only IMAP access to the job-alert inbox.

Read-only on purpose: the folder is opened with EXAMINE (select(readonly=True)) and messages are fetched with
BODY.PEEK, so nothing is flagged as read, moved, copied, deleted or expunged. There is no SMTP here either.
"""
from __future__ import annotations

import email
import imaplib
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email import policy
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Callable

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_HEADER_FIELDS = "MESSAGE-ID FROM DATE SUBJECT X-FORWARDED-FOR"
_UID_RE = re.compile(rb"UID (\d+)")
CHUNK = 100


class MailboxError(RuntimeError):
    """Connection, login or protocol problem. The message never contains the password."""


@dataclass
class Header:
    uid: str
    message_id: str
    sender: str          # address from the From header
    sender_domain: str
    subject: str
    date: datetime | None


def imap_date(d: date) -> str:
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}"   # not strftime: %b follows the locale


def _domain(addr: str) -> str:
    return addr.rsplit("@", 1)[-1].lower() if "@" in addr else ""


def _scrub(text: str, password: str) -> str:
    return text.replace(password, "***") if password else text


class Mailbox:
    def __init__(self, host: str, port: int, address: str, password: str, folder: str = "INBOX",
                 factory: Callable[..., Any] | None = None):
        self.host, self.port, self.address, self.password, self.folder = host, port, address, password, folder
        self.factory = factory or imaplib.IMAP4_SSL
        self.conn: Any = None

    def __enter__(self) -> Mailbox:
        try:
            self.conn = self.factory(self.host, self.port)
            self.conn.login(self.address, self.password)
            folder = f'"{self.folder}"' if " " in self.folder and not self.folder.startswith('"') else self.folder
            typ, data = self.conn.select(folder, readonly=True)
            if typ != "OK":
                raise MailboxError(f"cannot open folder {self.folder!r}: {data}")
        except MailboxError:
            self._close()
            raise
        except (imaplib.IMAP4.error, OSError) as e:
            self._close()
            hint = (" (check JOBMAIL_ADDRESS / JOBMAIL_APP_PASSWORD and that IMAP is enabled in Gmail)"
                    if isinstance(e, imaplib.IMAP4.error) else "")
            raise MailboxError(_scrub(f"IMAP {type(e).__name__}: {e}", self.password) + hint) from None
        return self

    def __exit__(self, *exc: object) -> None:
        self._close()

    def _close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            try:
                conn.logout()
            except Exception:  # noqa: BLE001 - closing a dead connection must not mask the real error
                pass

    def _uid(self, *args: str) -> list[Any]:
        try:
            typ, data = self.conn.uid(*args)
        except (imaplib.IMAP4.error, OSError) as e:
            raise MailboxError(_scrub(f"IMAP {type(e).__name__}: {e}", self.password)) from None
        if typ != "OK":
            raise MailboxError(_scrub(f"IMAP {args[0]} failed: {data}", self.password))
        return data

    def headers_since(self, since: date) -> list[Header]:
        """Headers of all messages received on or after `since`, oldest first."""
        found = self._uid("SEARCH", "SINCE", imap_date(since))
        uids = (found[0] or b"").split() if found else []
        out: list[Header] = []
        for i in range(0, len(uids), CHUNK):
            batch = b",".join(uids[i:i + CHUNK]).decode()
            data = self._uid("FETCH", batch, f"(UID BODY.PEEK[HEADER.FIELDS ({_HEADER_FIELDS})])")
            out += _parse_header_response(data)
        out.sort(key=lambda h: int(h.uid))
        return out

    def message(self, uid: str) -> EmailMessage:
        data = self._uid("FETCH", uid, "(BODY.PEEK[])")
        for item in data:
            if isinstance(item, tuple):
                return email.message_from_bytes(item[1], policy=policy.default)  # type: ignore[return-value]
        raise MailboxError(f"message {uid} came back empty")


def _parse_header_response(data: list[Any]) -> list[Header]:
    out: list[Header] = []
    for i, item in enumerate(data):
        if not isinstance(item, tuple):
            continue
        # The UID is in the tuple's first part, or (some servers) in the trailing bytes after the literal.
        m = _UID_RE.search(item[0])
        if m is None and i + 1 < len(data) and isinstance(data[i + 1], bytes):
            m = _UID_RE.search(data[i + 1])
        if m is None:
            continue
        msg = email.message_from_bytes(item[1], policy=policy.default)
        addr = parseaddr(str(msg.get("From", "")))[1]
        try:
            when = parsedate_to_datetime(str(msg["Date"])) if msg.get("Date") else None
            if when is not None and when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            when = when.astimezone(timezone.utc) if when else None
        except (TypeError, ValueError):
            when = None
        out.append(Header(
            uid=m.group(1).decode(),
            message_id=str(msg.get("Message-ID") or "").strip(),
            sender=addr,
            sender_domain=_domain(addr),
            subject=str(msg.get("Subject") or ""),
            date=when,
        ))
    return out
