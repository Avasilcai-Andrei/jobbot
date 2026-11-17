"""The daily email digest: jobs that came from job-alert emails and passed the rules.

An email has no description, so nobody has checked a job's requirements. Such jobs therefore never alert and are
never scored out of 100. They wait in the DB (status `digest`) and go out together once a day, at the first run at
or after `notify.digest.time`. A local model may add a coarse "title fit" (good / maybe / weak) that only orders the
list. If an ATS feed has the same role (cross-post logic), the ATS copy is the one that gets scored and may alert,
and the email copy is dropped here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import Config
from .db import DB
from .filters import Verdict
from .heartbeat import due
from .models import Job
from .notify import Digest, DigestItem, Notifier, sort_digest
from .scoring import MatchResult
from .text import crosspost_key

log = logging.getLogger(__name__)

META_KEY = "digest_sent_on"
GOOD_AT, MAYBE_AT = 75, 55   # the model's internal number -> a label; the number is never shown


def hint_for(score: int) -> str:
    return "good" if score >= GOOD_AT else "maybe" if score >= MAYBE_AT else "weak"


@dataclass
class Plan:
    keep: list[tuple[Job, Verdict]] = field(default_factory=list)
    dropped: list[tuple[Job, str, str | None]] = field(default_factory=list)   # job, reason, the job it duplicates


def select(cfg: Config, db: DB, entries: list[tuple[Job, Verdict]], now: datetime) -> Plan:
    """Drop the email copies that must not be shown: the same role exists on an ATS feed (which handles it), or
    another email in this batch already lists it. Normal-priority locations win over low-priority ones."""
    plan = Plan()
    since = now - timedelta(days=cfg.dedup.crosspost_window_days)
    seen: dict[str, Job] = {}
    unique = list({job.key: (job, v) for job, v in entries}.values())   # the same posting in several emails
    for job, v in sorted(unique, key=lambda jv: (jv[1].priority == "low", jv[0].title.lower())):
        key = crosspost_key(job.company, job.title)
        ats = db.find_ats_copy(key, since)
        if ats is not None:
            how = "scored normally" if ats["status"] != "filtered" else f"dropped there ({ats['reason']})"
            plan.dropped.append((job, f"same role on {ats['feed']}, {how}", f"{ats['source']}:{ats['job_id']}"))
        elif key in seen:
            plan.dropped.append((job, "same role listed by another email", f"{seen[key].source}:{seen[key].job_id}"))
        else:
            seen[key] = job
            plan.keep.append((job, v))
    return plan


def _item(job: Job, v: Verdict, hint: str | None) -> DigestItem:
    rule = v.location_rule
    return DigestItem(job=job, hint=hint, low_priority=rule.name if rule and rule.priority == "low" else None)


async def build(pipeline, plan: Plan, *, respect_control: bool) -> Digest:
    """Turn the kept jobs into a sorted digest, asking the model for title hints when it is free to answer."""
    cfg: Config = pipeline.cfg
    hints: list[str | None] = [None] * len(plan.keep)
    note: str | None = None
    if cfg.notify.digest.title_hint and plan.keep:
        results = await pipeline.score_all(plan.keep, respect_control=respect_control)
        for n, res in enumerate(results):
            if isinstance(res, MatchResult):
                hints[n] = hint_for(res.score)
            elif note is None:
                note = getattr(res, "reason", res)    # Deferred or the "unscored" reason: why the model didn't answer
        if note:
            note = str(note)[:200]
    items = [_item(j, v, h) for (j, v), h in zip(plan.keep, hints)]
    return Digest(items=sort_digest(items), day=pipeline.clock().astimezone().date().isoformat(), hint_note=note)


async def maybe_send(pipeline, now: datetime, now_local: datetime | None = None) -> bool:
    """Send today's digest if it is due and something is queued. True if one went out."""
    cfg: Config = pipeline.cfg
    dc, db = cfg.notify.digest, pipeline.db
    now_local = now_local or now.astimezone()
    if not dc.enabled or not due(now_local, dc.time, db.get_meta(META_KEY)):
        return False
    if pipeline.control.read().paused:
        return False
    queued = db.queued_for_digest()
    if not queued:
        return False

    entries: list[tuple[Job, Verdict]] = []
    for job in queued:   # re-run the rules: config or the posting's stored data may have changed since it was queued
        v = pipeline.rules.check(job)
        if v.ok:
            entries.append((job, v))
        else:
            db.update_job(job.key, status="filtered", reason=v.reason)
    plan = select(cfg, db, entries, now)
    for job, reason, of in plan.dropped:
        log.info("Digest: dropped %r @ %s: %s", job.title, job.company, reason)
        db.update_job(job.key, status="duplicate", duplicate_of=of, reason=reason)
    if not plan.keep:
        return False

    digest = await build(pipeline, plan, respect_control=True)
    if len(digest.items) > dc.max_items:
        digest.queued_more = len(digest.items) - dc.max_items
        digest.items = digest.items[:dc.max_items]
    sent_keys = {i.job.key for i in digest.items}
    try:
        await pipeline.notifier.send_digest(digest)
    except Exception as e:  # stays queued; tried again after the next run
        log.error("Could not send the email digest: %s", e)
        return False
    for job, _ in plan.keep:
        if job.key in sent_keys:
            db.update_job(job.key, status="digested", reason=f"in the email digest of {digest.day}")
    if not digest.queued_more:
        db.set_meta(META_KEY, now_local.date().isoformat())
    log.info("Sent the email digest: %d job(s)%s", len(digest.items),
             f", {digest.queued_more} left for the next one" if digest.queued_more else "")
    return True


async def preview(pipeline, *, use_llm: bool) -> tuple[Digest | None, Plan, list[tuple[Job, Verdict]]]:
    """For `jobbot digest`: what the digest would contain for the email jobs listed right now. Writes nothing."""
    email = [s for s in pipeline.sources if s.kind == "email"]
    fetched = await _fetch_all(pipeline, email)
    entries: list[tuple[Job, Verdict]] = []
    rejected: list[tuple[Job, Verdict]] = []
    for job in fetched:
        v = pipeline.rules.check(job)
        (entries if v.ok else rejected).append((job, v))
    plan = select(pipeline.cfg, pipeline.db, entries, pipeline.clock())
    if not plan.keep:
        return None, plan, rejected
    if use_llm and pipeline.cfg.notify.digest.title_hint:
        digest = await build(pipeline, plan, respect_control=False)
    else:
        digest = Digest(items=sort_digest([_item(j, v, None) for j, v in plan.keep]),
                        day=pipeline.clock().astimezone().date().isoformat(),
                        hint_note="the preview ran without --llm" if pipeline.cfg.notify.digest.title_hint else None)
    return digest, plan, rejected


async def _fetch_all(pipeline, sources) -> list[Job]:
    import asyncio
    jobs: list[Job] = []
    for source, (found, err) in zip(sources, await asyncio.gather(*(pipeline._fetch(s) for s in sources))):
        if err is not None:
            log.error("%s failed: %s", source.feed, err)
        else:
            jobs += found
    return jobs
