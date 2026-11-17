"""Notification channels: Discord webhook, Gmail SMTP email, and a chain that falls back or sends to both."""
from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import Literal, Protocol

import httpx

from .http import PoliteClient
from .models import Job

log = logging.getLogger(__name__)


@dataclass
class Alert:
    job: Job
    score: int | None             # None when running without the LLM stage
    reason: str
    missing: list[str]
    location_rule: str | None
    low_priority: bool
    first_seen_at: datetime
    unscored: str | None = None   # why there is no score (backend off/down, bad output)


@dataclass
class DigestItem:
    job: Job
    hint: str | None        # "good" | "maybe" | "weak": a model's reading of the title alone; None = not asked
    low_priority: str | None = None   # the lower-priority location rule that matched ("Bucharest", "Romania (country only)")


@dataclass
class Digest:
    """The daily list of jobs that came from job-alert emails. Nothing in it is an alert and none has a score."""
    items: list[DigestItem]
    day: str                    # "YYYY-MM-DD"
    hint_note: str | None = None  # why there are no hints, when they were wanted
    queued_more: int = 0        # jobs left for the next digest (over notify.digest.max_items)


class Notifier(Protocol):
    async def send_alert(self, alert: Alert) -> None: ...
    async def send_digest(self, digest: Digest) -> None: ...
    async def send_text(self, text: str) -> None: ...


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _color(score: int | None) -> int:
    if score is None:
        return 0x99AAB5
    if score >= 85:
        return 0x2ECC71
    if score >= 70:
        return 0xF1C40F
    return 0xE67E22


def discord_payload(alert: Alert) -> dict:
    job = alert.job
    location = job.location or "?"
    if alert.location_rule:
        location += f"\n({alert.location_rule}{', lower priority' if alert.low_priority else ''})"
    fields = [
        {"name": "Company", "value": _clip(job.company, 1024), "inline": True},
        {"name": "Location", "value": _clip(location, 1024), "inline": True},
        {"name": "Score", "value": f"{alert.score}/100" if alert.score is not None else "unscored", "inline": True},
    ]
    if alert.missing:
        fields.append({"name": "Missing", "value": _clip("\n".join(f"• {m}" for m in alert.missing), 1024)})
    if job.posted_at:
        lag = (alert.first_seen_at - job.posted_at).total_seconds() / 60
        fields.append({"name": "Posted", "value": f"<t:{int(job.posted_at.timestamp())}:R> (detected after {lag:.0f} min)"})
    fields.append({"name": "Mark", "value": f"`python -m jobbot mark {job.short_id} good` (or `bad`)"})
    title = ("[unscored] " if alert.unscored else "") + ("⚠️ " if alert.low_priority else "") + job.title
    return {
        "username": "jobbot",
        "allowed_mentions": {"parse": []},
        "embeds": [{
            "title": _clip(title, 256),
            "url": job.url,
            "description": _clip(alert.reason, 4096),
            "color": _color(alert.score),
            "fields": fields,
            "footer": {"text": f"{job.feed} · id {job.short_id}"},
        }],
    }


_PLATFORMS = {"linkedin": "LinkedIn", "indeed": "Indeed", "ejobs": "eJobs", "bestjobs": "BestJobs", "hipo": "Hipo",
              "glassdoor": "Glassdoor", "jobtogether": "JobTogether"}
_HINT_ORDER = {"good": 0, "maybe": 1, "weak": 2, None: 3}
DISCORD_LIMIT = 1900  # a message holds 2000 characters; keep some room


def _platform(job: Job) -> str:
    raw = str(job.extra.get("_email_platform") or "email")
    return _PLATFORMS.get(raw, raw)


def sort_digest(items: list[DigestItem]) -> list[DigestItem]:
    """Normal-priority locations first, then the model's hint, then by title. Stable and deterministic."""
    return sorted(items, key=lambda i: (i.low_priority is not None, _HINT_ORDER[i.hint], i.job.title.lower(), i.job.company.lower()))


def _digest_header(d: Digest) -> list[str]:
    platforms: dict[str, int] = {}
    for i in d.items:
        platforms[_platform(i.job)] = platforms.get(_platform(i.job), 0) + 1
    by = " · ".join(f"{n} {name}" for name, n in sorted(platforms.items(), key=lambda kv: -kv[1]))
    lines = [f"📬 jobbot email digest {d.day}: {len(d.items)} job{'' if len(d.items) == 1 else 's'} ({by})",
             "These came from job-alert emails and passed your rules. They are NOT alerts and have no score: "
             "the email has no description, so nobody has checked the requirements."]
    if any(i.hint for i in d.items):
        lines.append("Order: \"title fit\" is a local model's guess from the title alone (good / maybe / weak), "
                     "only a sorting hint.")
    elif d.hint_note:
        lines.append(f"No title-fit hints today: {d.hint_note}")
    return lines


def _digest_item(n: int, i: DigestItem, *, markdown: bool) -> str:
    j = i.job
    place = j.location or "?"
    what = f"{j.title} — {j.company} · {place}" + (f" ⚠️ low priority: {i.low_priority}" if i.low_priority else "")
    meta = " · ".join([_platform(j)] + ([f"title fit: {i.hint}"] if i.hint else []))
    if markdown:
        return f"**{n}.** {_clip(what, 300)}\n    {meta} · [open](<{j.url}>)"
    return f"{n}. {what}\n    {meta}\n    {j.url}"


def render_discord_digest(d: Digest) -> list[str]:
    """The digest as Discord messages of at most DISCORD_LIMIT characters (an item is never split)."""
    chunks: list[str] = ["\n".join(_digest_header(d))]
    for n, item in enumerate(d.items, 1):
        block = _digest_item(n, item, markdown=True)
        if len(block) > DISCORD_LIMIT:  # a tracking link too long for one message: say so instead of cutting it
            block = block.rsplit("[open]", 1)[0] + "(link too long for Discord: open the email)"
        if len(chunks[-1]) + len(block) + 1 > DISCORD_LIMIT:
            chunks.append(block)
        else:
            chunks[-1] += "\n" + block
    if d.queued_more:
        tail = f"…and {d.queued_more} more, in tomorrow's digest."
        if len(chunks[-1]) + len(tail) + 1 > DISCORD_LIMIT:
            chunks.append(tail)
        else:
            chunks[-1] += "\n" + tail
    return chunks


def render_email_digest(d: Digest) -> tuple[str, str]:
    """(subject, plain-text body)."""
    parts = _digest_header(d) + [_digest_item(n, i, markdown=False) for n, i in enumerate(d.items, 1)]
    if d.queued_more:
        parts.append(f"...and {d.queued_more} more, in tomorrow's digest.")
    return f"[jobbot] Email digest {d.day}: {len(d.items)} job{'' if len(d.items) == 1 else 's'}", "\n\n".join(parts)


class NotifyError(Exception):
    pass


class DiscordNotifier:
    def __init__(self, webhook_url: str, http: PoliteClient):
        self.url = webhook_url
        self.http = http

    async def _post(self, payload: dict) -> None:
        try:
            await self.http.post(self.url, json=payload)
        except httpx.HTTPStatusError as e:
            # Don't let the webhook URL (it contains the token) end up in logs.
            raise NotifyError(f"Discord webhook returned HTTP {e.response.status_code}: "
                              f"{e.response.text[:200]}") from None
        except httpx.TransportError as e:
            raise NotifyError(f"Discord webhook unreachable: {type(e).__name__}") from None

    async def send_alert(self, alert: Alert) -> None:
        await self._post(discord_payload(alert))

    async def send_digest(self, digest: Digest) -> None:
        for chunk in render_discord_digest(digest):
            await self._post({"username": "jobbot", "content": chunk, "allowed_mentions": {"parse": []}})

    async def send_text(self, text: str) -> None:
        await self._post({"username": "jobbot", "content": _clip(text, 2000), "allowed_mentions": {"parse": []}})


def email_for_alert(alert: Alert) -> tuple[str, str]:
    """(subject, plain-text body)."""
    job = alert.job
    score = f"{alert.score}/100" if alert.score is not None else "unscored"
    prefix = "[unscored] " if alert.unscored else ""
    location = job.location or "?"
    if alert.location_rule:
        location += f" ({alert.location_rule}{', lower priority' if alert.low_priority else ''})"
    lines = [f"{job.title} @ {job.company}", f"Location: {location}", f"Score: {score}", "", alert.reason]
    if alert.missing:
        lines += ["", "Missing:", *(f"  - {m}" for m in alert.missing)]
    if job.posted_at:
        lines += ["", f"Posted: {job.posted_at:%Y-%m-%d %H:%M} UTC"]
    lines += ["", job.url, "", f"Mark it: python -m jobbot mark {job.short_id} good|bad [note]",
              "", f"-- jobbot ({job.feed}, id {job.short_id})"]
    subject = f"[jobbot] {prefix}{score}: {job.title} @ {job.company}"
    return _clip(subject.replace("\n", " "), 200), "\n".join(lines)


class EmailNotifier:
    """Gmail (or any SMTP server with implicit TLS) using an app password from .env."""

    def __init__(self, user: str, password: str, to: str, host: str = "smtp.gmail.com", port: int = 465,
                 timeout: float = 30, smtp_factory=smtplib.SMTP_SSL):
        self.user, self.password, self.to = user, password, to
        self.host, self.port, self.timeout = host, port, timeout
        self._smtp_factory = smtp_factory

    def _send_sync(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.user, self.to, subject
        msg.set_content(body)
        with self._smtp_factory(self.host, self.port, timeout=self.timeout,
                                context=ssl.create_default_context()) as smtp:
            smtp.login(self.user, self.password)
            smtp.send_message(msg)

    async def _send(self, subject: str, body: str) -> None:
        try:
            await asyncio.to_thread(self._send_sync, subject, body)
        except smtplib.SMTPAuthenticationError as e:
            # Never include the password; the server's code is enough to diagnose.
            raise NotifyError(f"SMTP login failed for {self.user} (code {e.smtp_code}); check SMTP_APP_PASSWORD") \
                from None
        except (smtplib.SMTPException, OSError) as e:
            raise NotifyError(f"Email send failed: {type(e).__name__}: {str(e)[:150]}") from None

    async def send_alert(self, alert: Alert) -> None:
        await self._send(*email_for_alert(alert))

    async def send_digest(self, digest: Digest) -> None:
        await self._send(*render_email_digest(digest))

    async def send_text(self, text: str) -> None:
        first = text.strip().splitlines()[0] if text.strip() else "notice"
        await self._send(_clip(f"[jobbot] {first}", 200), text)


class NotifierChain:
    """`fallback`: try each channel in order until one succeeds. `both`: send to every channel; it counts as
    delivered if at least one succeeds. Raises NotifyError only when every channel failed."""

    def __init__(self, notifiers: list[tuple[str, Notifier]], mode: Literal["fallback", "both"] = "fallback"):
        self.notifiers = notifiers
        self.mode = mode

    async def _deliver(self, method: str, arg: object) -> None:
        errors = []
        delivered = False
        for i, (name, n) in enumerate(self.notifiers):
            try:
                await getattr(n, method)(arg)
                delivered = True
                if self.mode == "fallback":
                    return
            except Exception as e:
                errors.append(f"{name}: {e}")
                more = self.mode == "fallback" and i < len(self.notifiers) - 1
                log.warning("%s failed: %s%s", name, e, "; trying the next channel" if more else "")
        if not delivered:
            raise NotifyError("; ".join(errors) or "no notification channel configured")

    async def send_alert(self, alert: Alert) -> None:
        await self._deliver("send_alert", alert)

    async def send_digest(self, digest: Digest) -> None:
        await self._deliver("send_digest", digest)

    async def send_text(self, text: str) -> None:
        await self._deliver("send_text", text)


class ConsoleNotifier:
    """Used when no webhook is configured, and by `dry-run`."""

    async def send_alert(self, alert: Alert) -> None:
        score = f"{alert.score}/100" if alert.score is not None else "[unscored]"
        prio = " [lower priority]" if alert.low_priority else ""
        print(f"\n=== ALERT {score}{prio}: {alert.job.title} @ {alert.job.company}  [id {alert.job.short_id}]\n"
              f"    {alert.job.location}\n    {alert.job.url}\n    {alert.reason}")
        if alert.missing:
            print("    missing: " + "; ".join(alert.missing))

    async def send_digest(self, digest: Digest) -> None:
        for n, chunk in enumerate(render_discord_digest(digest), 1):
            print(f"\n=== DIGEST message {n} ({len(chunk)} chars) ===\n{chunk}")

    async def send_text(self, text: str) -> None:
        print(f"\n=== NOTICE: {text}")
