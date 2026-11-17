"""One pipeline run: fetch -> dedup -> rule filter -> LLM match -> notify, plus feed health."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from typing import Callable, TypeVar

import httpx

from .config import Config
from .control import Control
from .db import DB, utcnow
from .filters import RuleFilter, Verdict
from . import ollama_ctl
from .guard import GpuGuard
from .http import BadResponse, PoliteClient, safe_url
from .models import Job
from .notify import Alert, Notifier
from .scoring import MatchResult, Scorer, ScoreFailed, ScoreUnavailable, ScoreUnreachable
from .sources.base import Source
from .text import crosspost_key

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
T = TypeVar("T")


@dataclass
class RunStats:
    feeds_ok: int = 0
    feeds_failed: int = 0
    feeds_skipped: int = 0
    fetched: int = 0
    seeded: int = 0
    new: int = 0
    filtered: int = 0
    duplicates: int = 0
    passed_rules: int = 0  # passed stage 1 and cross-post dedup, i.e. went on to scoring
    scored: int = 0        # got an LLM score (the rest of passed_rules went out [unscored] or failed)
    scored_low: int = 0
    rescored: int = 0     # scored a second time because the first score was near the threshold
    notified: int = 0
    unscored: int = 0
    retry_later: int = 0
    held: int = 0           # passed the rules but left waiting for scoring (paused / deferred / GPU busy)
    digest_queued: int = 0  # email jobs that passed the rules and now wait for the daily digest (never alerted)
    auto_deferred: int = 0  # 1 if the low-free-VRAM guard deferred scoring this run
    paused: int = 0         # 1 if the run was skipped because the bot is paused (not stored in `runs`)

    def summary(self) -> str:
        return ", ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


@dataclass
class Deferred:
    """score_all's answer for a job that must wait: scoring is paused / deferred / the GPU is busy."""
    reason: str
    auto: bool = False


def describe_error(err: BaseException) -> str:
    if isinstance(err, httpx.HTTPStatusError):
        return f"HTTP {err.response.status_code} from {safe_url(str(err.request.url))}"
    if isinstance(err, BadResponse):  # already bounded; the body preview alone is up to 300 characters
        return f"BadResponse: {err}"
    return f"{type(err).__name__}: {err}"[:300]


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _by_kind(items: list[T], key: Callable[[T], Source]) -> dict[str, list[T]]:
    groups: dict[str, list[T]] = {}
    for item in items:
        groups.setdefault(key(item).kind, []).append(item)
    return groups


def error_kind(err: BaseException) -> str:
    """A short label that is the same for the same kind of failure on different feeds."""
    if isinstance(err, httpx.HTTPStatusError):
        return f"HTTP {err.response.status_code}"
    if isinstance(err, BadResponse):
        return f"HTTP {err.status_code} {err.content_type.split(';')[0] or 'no content-type'} (not JSON)"
    return type(err).__name__


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        db: DB,
        http: PoliteClient,
        sources: list[Source],
        notifier: Notifier,
        scorer: Scorer | None,
        clock: Callable[[], datetime] = utcnow,
        scorer_off_reason: str | None = None,
        control: Control | None = None,
        guard: GpuGuard | None = None,
        ollama_launcher: Callable[[], str | None] = ollama_ctl.launch_server,
    ):
        self.cfg = cfg
        self.db = db
        self.http = http
        self.sources = sources
        self.by_feed = {s.feed: s for s in sources}
        for s in sources:
            s.bind(db)
        self.notifier = notifier
        self.scorer = scorer
        self.scorer_off_reason = scorer_off_reason
        self.rules = RuleFilter(cfg)
        self.clock = clock
        self.control = control or Control(db, cfg, clock)
        self.guard = guard or GpuGuard(cfg, db, own_model=getattr(scorer, "model", None), clock=clock)
        self.ollama_launcher = ollama_launcher
        self._ollama_start_tried = False   # one start attempt per outage; reset once the server answers again

    # --- fetching & health ------------------------------------------------

    async def _fetch(self, source: Source) -> tuple[list[Job], BaseException | None]:
        try:
            return await source.fetch(self.http), None
        except Exception as e:  # one broken source must not break the run
            return [], e

    async def _notify_text(self, text: str) -> bool:
        if self.control.read().paused:  # no health warnings while paused
            log.info("Paused: not sending notice %r", text[:60])
            return False
        try:
            await self.notifier.send_text(text)
            return True
        except Exception as e:
            log.error("Could not send notice: %s", e)
            return False

    def _feed_failed(self, source: Source, err: BaseException, now: datetime) -> bool:
        """Record a failure. True if the feed now needs a warning (sent grouped by `_send_health_notices`)."""
        st = self.db.feed_state(source.feed)
        st.consecutive_failures += 1
        st.last_attempt_at = now
        st.last_error = describe_error(err)
        threshold = self.cfg.health.failure_threshold
        log.warning("%s failed (%d in a row): %s", source.feed, st.consecutive_failures, st.last_error)
        if st.consecutive_failures >= threshold:
            # Back off across runs: 1, 2, 4... intervals, at most an hour between attempts.
            interval = timedelta(minutes=self.cfg.schedule.interval_minutes)
            st.next_attempt_at = now + min(interval * 2 ** (st.consecutive_failures - threshold), timedelta(hours=1))
        self.db.save_feed_state(st)
        return st.consecutive_failures >= threshold and not st.warned

    def _feed_ok(self, source: Source, now: datetime) -> bool:
        """Record a success. True if a warning had been sent, so a recovery notice is due."""
        st = self.db.feed_state(source.feed)
        was_warned = st.warned
        st.consecutive_failures = 0
        st.warned = False
        st.next_attempt_at = None
        st.last_error = None
        st.last_attempt_at = st.last_success_at = now
        self.db.save_feed_state(st)
        return was_warned

    async def _send_health_notices(self, failing: list[tuple[Source, BaseException]],
                                   recovered: list[Source]) -> None:
        """One warning per source kind for the feeds that crossed the failure threshold this run, and one
        recovery notice per kind. A single feed keeps the one-feed wording."""
        for kind, group in _by_kind(failing, key=lambda item: item[0]).items():
            states = [self.db.feed_state(s.feed) for s, _ in group]
            if len(group) == 1:
                st = states[0]
                text = (f"⚠️ jobbot: source `{st.feed}` has failed {st.consecutive_failures} runs in a row.\n"
                        f"Last error: {st.last_error}")
            else:
                counts = sorted({st.consecutive_failures for st in states})
                runs = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
                kinds = sorted({error_kind(e) for _, e in group})
                same = (f"All with the same kind of error ({kinds[0]})." if len(kinds) == 1
                        else f"Errors: {', '.join(kinds)}.")
                text = (f"⚠️ jobbot: {len(group)} {kind} sources have failed {runs} runs in a row: "
                        f"{', '.join(f'`{s.name}`' for s, _ in group)}.\n{same}\n"
                        f"Last error (`{states[0].feed}`): {_clip(states[0].last_error or '', 900)}")
            if await self._notify_text(text):
                for st in states:
                    st.warned = True
                    self.db.save_feed_state(st)
        for kind, group in _by_kind(recovered, key=lambda s: s).items():
            if len(group) == 1:
                await self._notify_text(f"✅ jobbot: source `{group[0].feed}` is working again.")
            else:
                await self._notify_text(f"✅ jobbot: {len(group)} {kind} sources are working again: "
                                        f"{', '.join(f'`{s.name}`' for s in group)}.")

    def _ingest(self, source: Source, jobs: list[Job], now: datetime, stats: RunStats) -> None:
        st = self.db.feed_state(source.feed)
        seeding = st.seeded_at is None
        for job in jobs:
            stats.fetched += 1
            if self.db.job_exists(job.key):
                continue
            inserted = self.db.insert_job(
                job,
                status="seeded" if seeding else "new",
                first_seen_at=now,
                crosspost_key=crosspost_key(job.company, job.title),
            )
            if inserted:
                if seeding:
                    stats.seeded += 1
                else:
                    stats.new += 1
        if seeding:
            st.seeded_at = now
            self.db.save_feed_state(st)
            log.info("First run for %s: marked %d existing postings as seen (no alerts)", source.feed, len(jobs))

    # --- processing new jobs ----------------------------------------------

    def _retry_later(self, job: Job, why: str, stats: RunStats) -> None:
        attempts = self.db.attempts(job.key) + 1
        if attempts >= MAX_ATTEMPTS:
            log.error("Giving up on %s after %d attempts: %s", job.url, attempts, why)
            self.db.update_job(job.key, status="error", attempts=attempts, reason=why)
        else:
            log.warning("Will retry %s next run (%d/%d): %s", job.url, attempts, MAX_ATTEMPTS, why)
            self.db.update_job(job.key, attempts=attempts, reason=why)
        stats.retry_later += 1

    async def _load_details(self, job: Job, store: bool = True) -> BaseException | None:
        source = self.by_feed.get(job.feed)
        try:
            if source is None:  # feed removed from config since the job was stored
                job.description = ""
            else:
                await source.fetch_details(self.http, job)
                if job.description is None:
                    job.description = ""
            if store:  # details can also refine location, remote flag and metadata
                self.db.update_job(job.key, description=job.description, location=job.location, url=job.url,
                                   remote=None if job.remote is None else int(job.remote),
                                   extra=json.dumps(job.extra, ensure_ascii=False))
            return None
        except Exception as e:
            return e

    def threshold(self, verdict: Verdict) -> int:
        rule = verdict.location_rule
        return rule.min_score if rule and rule.min_score is not None else self.cfg.notify.min_score

    @staticmethod
    def _location_note(verdict: Verdict) -> str | None:
        rule = verdict.location_rule
        if rule is None:
            return None
        return f"{rule.name}, lower priority" if rule.priority == "low" else rule.name

    async def _scoring_gate(self) -> Deferred | None:
        """Why scoring must wait right now, or None. Checked once per batch, before the model is loaded."""
        block = self.control.scoring_block()
        if block:
            return Deferred(block)
        if self.scorer is not None and self.scorer.uses_gpu:
            busy = await self.guard.check()
            if busy.busy:
                return Deferred(busy.describe(), auto=True)
        return None

    async def _try_start_ollama(self, problem: str) -> str | None:
        """Ollama doesn't answer: start it once (the tray app, as the autostart wrapper does) and wait for it.
        Returns the problem to report, or None when the server answers now. At most one attempt per outage."""
        o = self.cfg.llm.ollama
        if not o.auto_start or self._ollama_start_tried:
            return problem
        self._ollama_start_tried = True
        log.warning("%s. Trying to start it (waiting up to %.0f s).", problem, o.auto_start_wait_seconds)
        async with httpx.AsyncClient(base_url=o.url.rstrip("/")) as client:
            ok, note = await ollama_ctl.start_and_wait(client, o.auto_start_wait_seconds, self.ollama_launcher)
        log.warning("Ollama start attempt: %s", note)
        if ok:
            return await self.scorer.check_available()
        return f"{problem}; {note}"

    async def score_all(self, items: list[tuple[Job, Verdict]], respect_control: bool = True
                        ) -> list[MatchResult | str | Deferred]:
        """Score each job. A str result means "unscored", and says why; a Deferred means "score it later".
        Never raises. `respect_control=False` (dry-run) ignores pause/defer and the VRAM guard."""
        if not items:
            return []
        if self.scorer is None:
            reason = self.scorer_off_reason or "scoring is off"
            log.info("Not scoring %d job(s): %s; sending rule matches as [unscored]", len(items), reason)
            return [reason] * len(items)
        if respect_control and (deferred := await self._scoring_gate()):
            log.info("Not scoring %d job(s) now: %s", len(items), deferred.reason)
            return [deferred] * len(items)
        problem = await self.scorer.check_available()
        if problem and getattr(self.scorer, "unreachable", False) and respect_control:
            problem = await self._try_start_ollama(problem)
        if not problem:
            self._ollama_start_tried = False
        if problem:
            if getattr(self.scorer, "unreachable", False):
                # The server isn't answering (even after one start attempt). That usually heals, so hold the jobs;
                # _hold_or_release sends them [unscored] once they have waited control.defer_max_hold_minutes.
                log.warning("Scoring unavailable this run: %s. Holding %d job(s); they go out [unscored] after %s min.",
                            problem, len(items), self.cfg.control.defer_max_hold_minutes)
                return [Deferred(problem)] * len(items)
            log.warning("Scoring unavailable this run: %s. Sending %d rule match(es) as [unscored].",
                        problem, len(items))
            return [problem] * len(items)

        down: str | Deferred | None = None

        async def one(job: Job, v: Verdict) -> MatchResult | str | Deferred:
            nonlocal down
            if down:
                return down
            if respect_control and (block := self.control.scoring_block()):  # paused / deferred mid-run
                return Deferred(block)
            try:
                with self.guard.scoring():  # our own GPU use must not count as "someone else is using it"
                    return await self.scorer.score_stable(job, self._location_note(v), self.threshold(v),
                                                          self.cfg.llm.rescore_margin)
            except ScoreUnavailable as e:
                if down is None:
                    down = Deferred(str(e)) if isinstance(e, ScoreUnreachable) else str(e)
                    log.warning("Scoring became unavailable: %s. Remaining jobs this run %s.", e,
                                "are held for the next run" if isinstance(down, Deferred) else "go out [unscored]")
                return down
            except ScoreFailed as e:
                log.warning("Could not score %r: %s; sending it [unscored]", job.title, e)
                return f"scoring failed: {e}"
            except Exception as e:  # never drop a job because the scorer broke
                log.exception("Unexpected scoring error for %r", job.title)
                return f"scoring error: {type(e).__name__}"

        return list(await asyncio.gather(*(one(j, v) for j, v in items)))

    async def _process_new(self, now: datetime, stats: RunStats) -> None:
        pending = self.db.jobs_with_status("new")
        held = self.db.jobs_with_status("pending")  # passed the rules earlier, still waiting for a score
        if not pending and not held:
            return

        # Stage 1a: title + location, no network.
        survivors: list[Job] = []
        for job in pending:
            v = self.rules.prefilter(job)
            if v.ok:
                survivors.append(job)
            else:
                self.db.update_job(job.key, status="filtered", reason=v.reason)
                stats.filtered += 1

        # Fetch descriptions only for jobs that might matter.
        need = [j for j in survivors if j.description is None]
        errors = await asyncio.gather(*(self._load_details(j) for j in need))
        failed = {j.key: e for j, e in zip(need, errors) if e is not None}

        # Stage 1b: description-based rules.
        checked: list[tuple[Job, Verdict]] = []
        for job in survivors:
            if job.key in failed:
                self._retry_later(job, f"details fetch failed: {describe_error(failed[job.key])}", stats)
                continue
            v = self.rules.check(job)
            if v.ok:
                checked.append((job, v))
            else:
                log.info("Dropped %r @ %s after reading the description: %s", job.title, job.company, v.reason)
                self.db.update_job(job.key, status="filtered", reason=v.reason)
                stats.filtered += 1

        # Cross-post dedup, high-priority locations first so they win over the same role elsewhere.
        # Email jobs never alert and are never scored: they queue for the daily digest (see digest.py), after the
        # ATS jobs of this batch have been deduplicated, so that an ATS copy always wins over an email copy.
        from_email = [jv for jv in checked if jv[0].source == "email"]
        checked = [jv for jv in checked if jv[0].source != "email"]
        checked.sort(key=lambda jv: (jv[1].priority != "high", jv[0].posted_at.timestamp() if jv[0].posted_at else 0))
        since = now - timedelta(days=self.cfg.dedup.crosspost_window_days)
        to_score: list[tuple[Job, Verdict]] = []
        for job, v in checked:
            dup = self.db.find_crosspost(crosspost_key(job.company, job.title), job.key, since)
            if dup is not None:
                self.db.update_job(job.key, status="duplicate", duplicate_of=f"{dup['source']}:{dup['job_id']}",
                                   reason=f"cross-post of {dup['feed']} job {dup['job_id']}")
                stats.duplicates += 1
                continue
            self.db.update_job(job.key, passed_rules=1, priority=v.priority)
            to_score.append((job, v))
        stats.passed_rules += len(to_score)

        for job, v in sorted(from_email, key=lambda jv: (jv[1].priority == "low", jv[0].title.lower())):
            key = crosspost_key(job.company, job.title)
            ats = self.db.find_ats_copy(key, since)
            dup = ats or self.db.find_email_crosspost(key, job.key, since)
            if dup is not None:
                self.db.update_job(job.key, status="duplicate", duplicate_of=f"{dup['source']}:{dup['job_id']}",
                                   reason=f"cross-post of {dup['feed']} job {dup['job_id']}")
                stats.duplicates += 1
                continue
            self.db.update_job(job.key, status="digest", passed_rules=1, priority=v.priority,
                               reason="queued for the daily email digest")
            stats.digest_queued += 1

        # Jobs held back earlier go first. Their description is stored, so the rules re-run without network.
        carried: list[tuple[Job, Verdict]] = []
        for job in held:
            v = self.rules.check(job)
            if v.ok:
                carried.append((job, v))
            else:
                self.db.update_job(job.key, status="filtered", reason=v.reason)
                stats.filtered += 1
        to_score = carried + to_score

        # Stage 2: score (the scorer limits its own concurrency).
        results = await self.score_all(to_score)
        stats.auto_deferred += any(isinstance(r, Deferred) and r.auto for r in results)

        for (job, v), res in zip(to_score, results):
            if isinstance(res, Deferred):
                res = self._hold_or_release(job, res, now, stats)
                if res is None:
                    continue
            rule = v.location_rule
            threshold = self.threshold(v)
            common = dict(job=job, location_rule=rule.name if rule else None,
                          low_priority=bool(rule and rule.priority == "low"),
                          first_seen_at=self._first_seen(job, now))
            if isinstance(res, MatchResult):
                match_json = json.dumps(res.model_dump(), ensure_ascii=False)
                stats.scored += 1
                stats.rescored += res.scores is not None
                twice = f" (scored twice: {res.scores[0]} and {res.scores[1]})" if res.scores else ""
                if res.score < threshold:
                    self.db.update_job(job.key, status="scored", score=res.score, match_json=match_json,
                                       reason=f"score {res.score} < {threshold}{twice}")
                    stats.scored_low += 1
                    continue
                reason = f"{res.reason} Scored twice near the threshold: {res.scores[0]} and {res.scores[1]}." \
                    if res.scores else res.reason
                alert = Alert(score=res.score, reason=reason, missing=res.missing_requirements, **common)
            else:
                match_json = None
                alert = Alert(score=None, reason=f"Matched the rule filter as {v.role}. Not scored: {res}.",
                              missing=[], unscored=res, **common)
            try:
                await self.notifier.send_alert(alert)
            except Exception as e:
                self._retry_later(job, f"notify failed: {e}", stats)
                continue
            self.db.update_job(job.key, status="notified", notified_at=self.clock(), score=alert.score,
                               match_json=match_json, reason=f"unscored: {res}" if alert.unscored else None)
            stats.notified += 1
            stats.unscored += alert.unscored is not None
            log.info("Notified: %s @ %s (%s)", job.title, job.company,
                     f"score {alert.score}" if alert.score is not None else "unscored")

    def _hold_or_release(self, job: Job, d: Deferred, now: datetime, stats: RunStats) -> str | None:
        """Keep a deferred job `pending`, or, once it has waited longer than control.defer_max_hold_minutes, return
        the "unscored" reason so it goes out as [unscored] and nothing urgent is lost."""
        hold = self.cfg.control.defer_max_hold_minutes
        waited = self.clock() - self._first_seen(job, now)
        if hold is not None and waited >= timedelta(minutes=hold):
            return f"scoring was deferred for over {hold:g} min ({d.reason})"
        self.db.update_job(job.key, status="pending", reason=f"waiting for scoring: {d.reason}")
        stats.held += 1
        return None

    def _first_seen(self, job: Job, default: datetime) -> datetime:
        row = self.db.conn.execute(
            "SELECT first_seen_at FROM jobs WHERE source=? AND job_id=?", job.key
        ).fetchone()
        return datetime.fromisoformat(row["first_seen_at"]) if row else default

    # --- entry point ------------------------------------------------------

    async def drain_pending(self) -> RunStats:
        """Score the jobs that were waiting, without fetching (used when a deferral ends between runs)."""
        stats = RunStats()
        await self._process_new(self.clock(), stats)
        if stats.notified or stats.scored:
            log.info("Caught up on waiting jobs: %s", stats.summary())
        return stats

    async def unload_model(self) -> None:
        if self.scorer is not None:
            await self.scorer.unload()

    async def run_once(self) -> RunStats:
        now = self.clock()
        stats = RunStats()
        state = self.control.begin_run(now)
        if state.paused:
            log.info("Paused: skipping this run (resume with `python -m jobbot resume`)")
            stats.paused = 1
            return stats
        due: list[Source] = []
        for s in self.sources:
            st = self.db.feed_state(s.feed)
            if st.next_attempt_at and st.next_attempt_at > now + timedelta(seconds=30):
                stats.feeds_skipped += 1
                log.info("Skipping %s until %s (backing off after errors)", s.feed, st.next_attempt_at)
                continue
            due.append(s)

        results = await asyncio.gather(*(self._fetch(s) for s in due))
        failing: list[tuple[Source, BaseException]] = []
        recovered: list[Source] = []
        for source, (jobs, err) in zip(due, results):
            if err is not None:
                if self._feed_failed(source, err, now):
                    failing.append((source, err))
                stats.feeds_failed += 1
                continue
            if self._feed_ok(source, now):
                recovered.append(source)
            stats.feeds_ok += 1
            self._ingest(source, jobs, now, stats)
            source.after_ingest(now)
        await self._send_health_notices(failing, recovered)

        await self._process_new(now, stats)
        self.db.record_run(now, self.clock(), asdict(stats))
        return stats


@dataclass
class Evaluation:
    job: Job
    verdict: Verdict
    match: MatchResult | None = None
    error: str | None = None


async def evaluate_current(pipeline: Pipeline, use_llm: bool) -> list[Evaluation]:
    """For `dry-run`: run stage 1 (and optionally stage 2) over every job currently listed.
    Nothing is written to the jobs table and nothing is sent."""
    results = await asyncio.gather(*(pipeline._fetch(s) for s in pipeline.sources))
    out: list[Evaluation] = []
    for source, (jobs, err) in zip(pipeline.sources, results):
        if err is not None:
            log.error("%s failed: %s", source.feed, describe_error(err))
            continue
        for job in jobs:
            v = pipeline.rules.prefilter(job)
            out.append(Evaluation(job, v))
    survivors = [e for e in out if e.verdict.ok]
    errors = await asyncio.gather(*(pipeline._load_details(e.job, store=False)
                                    for e in survivors if e.job.description is None))
    for err in filter(None, errors):
        log.warning("Description fetch failed: %s", describe_error(err))
    for e in survivors:
        e.verdict = pipeline.rules.check(e.job)
    if use_llm:
        # Email jobs are never scored (they go to the digest; `jobbot digest` previews it).
        passed = [e for e in survivors if e.verdict.ok and e.job.source != "email"]
        scored = await pipeline.score_all([(e.job, e.verdict) for e in passed], respect_control=False)
        for e, res in zip(passed, scored):
            if isinstance(res, MatchResult):
                e.match = res
            else:
                e.error = res
    return out
