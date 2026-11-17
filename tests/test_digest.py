"""Email jobs go to a daily digest (never alert), cross-source dedup, Ollama start-and-hold, offline latency."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from jobbot import digest as digest_mod, ollama_ctl
from jobbot.db import DB
from jobbot.filters import RuleFilter
from jobbot.notify import Digest, DigestItem, ConsoleNotifier, render_discord_digest, render_email_digest
from jobbot.pipeline import Pipeline
from jobbot.report import bot_was_offline, build_report
from jobbot.scoring import ScoreUnreachable

from .conftest import make_config, make_job, mock_client
from .test_pipeline import Clock, FakeScorer, FakeSource, RecordingNotifier, gh, run, status


class DigestNotifier(RecordingNotifier):
    def __init__(self):
        super().__init__()
        self.digests = []

    async def send_digest(self, digest):
        if self.fail:
            raise RuntimeError("discord down")
        self.digests.append(digest)


def em(job_id, **kw):
    """A job as the email source returns it: no description, tagged as from email."""
    kw.setdefault("source", "email")
    kw.setdefault("feed", "email:jobmail")
    kw.setdefault("description", "")
    kw.setdefault("url", f"https://example.com/e/{job_id}")
    kw.setdefault("extra", {"_from_email": "1", "_email_platform": "linkedin"})
    kw.setdefault("posted_at", None)
    return make_job(job_id, **kw)


@pytest.fixture
def env(cfg):
    clock = Clock()
    clock.now = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    db = DB(cfg.db_path)
    ats, mail = FakeSource(), FakeSource("jobmail")
    scorer, notifier = FakeScorer(), DigestNotifier()
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [ats, mail], notifier, scorer, clock=clock)
    run(p)   # seeds both feeds
    clock.tick()
    yield p, ats, mail, scorer, notifier, clock, db
    db.close()


# --- email jobs never alert; they queue for the digest ---------------------------------------------------------

def test_email_job_is_queued_not_alerted_or_scored(env):
    p, ats, mail, scorer, notifier, clock, db = env
    mail.jobs = [em("1", title="Junior Data Engineer")]
    stats = run(p)
    assert stats.digest_queued == 1 and stats.notified == 0 and stats.scored == 0
    assert scorer.calls == [] and notifier.alerts == []
    row = status(db, "1", "email")
    assert row["status"] == "digest" and row["passed_rules"] == 1 and row["notified_at"] is None


def test_email_job_still_needs_to_pass_the_rules(env):
    p, ats, mail, scorer, notifier, clock, db = env
    mail.jobs = [em("1", title="Senior Data Engineer"), em("2", title="Office Manager")]
    run(p)
    assert status(db, "1", "email")["status"] == "filtered" and status(db, "2", "email")["status"] == "filtered"


def test_bare_romania_is_kept_for_email_jobs_only_and_marked_low_priority(env, cfg):
    p, ats, mail, scorer, notifier, clock, db = env
    mail.jobs = [em("1", title="Junior Data Engineer", location="Romania")]
    ats.jobs = [gh("2", title="Junior Data Scientist", location="Romania", description="Python. On-site in Romania.")]
    run(p)
    assert status(db, "1", "email")["status"] == "digest" and status(db, "1", "email")["priority"] == "low"
    assert status(db, "2")["status"] == "filtered"   # an ATS job still needs a real location match
    rules = RuleFilter(cfg)
    assert rules.prefilter(em("3", title="Junior Data Engineer", location="Cluj-Napoca Metropolitan Area")).priority == "high"
    assert not rules.prefilter(em("4", title="Junior Data Engineer", location="Germany")).ok


# --- cross-source dedup -------------------------------------------------------------------------------------------

def test_ats_copy_wins_when_it_came_first(env):
    p, ats, mail, scorer, notifier, clock, db = env
    ats.jobs = [gh("1", title="Junior Data Engineer", company="Initech")]
    run(p)
    assert len(notifier.alerts) == 1
    clock.tick()
    mail.jobs = [em("9", title="Junior Data Engineer", company="Initech")]
    stats = run(p)
    assert stats.duplicates == 1 and stats.digest_queued == 0
    row = status(db, "9", "email")
    assert row["status"] == "duplicate" and row["duplicate_of"] == "fake:1"
    assert len(notifier.alerts) == 1   # never twice


def test_ats_copy_that_arrives_later_is_scored_and_the_queued_email_copy_is_dropped_from_the_digest(env):
    p, ats, mail, scorer, notifier, clock, db = env
    mail.jobs = [em("9", title="Junior Data Engineer", company="Initech"), em("8", title="Python Backend Intern", company="Other")]
    run(p)
    assert status(db, "9", "email")["status"] == "digest"
    clock.tick()
    ats.jobs = [gh("1", title="Junior Data Engineer", company="Initech")]
    run(p)
    assert [a.job.job_id for a in notifier.alerts] == ["1"]                 # the ATS copy alerts normally
    clock.now = datetime(2026, 11, 8, 18, 5).astimezone().astimezone(timezone.utc)  # 18:05 local
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    assert [i.job.job_id for i in notifier.digests[0].items] == ["8"]
    assert status(db, "9", "email")["status"] == "duplicate"
    assert status(db, "8", "email")["status"] == "digested"


def test_ats_copy_rejected_by_its_description_also_drops_the_email_copy(env):
    p, ats, mail, scorer, notifier, clock, db = env
    ats.jobs = [gh("1", title="Junior Data Engineer", company="Initech", extra={"_desc": "Requires 5+ years of experience."})]
    run(p)
    assert status(db, "1")["status"] == "filtered"
    clock.tick()
    mail.jobs = [em("9", title="Junior Data Engineer", company="Initech")]
    run(p)
    assert status(db, "9", "email")["status"] == "duplicate"


def test_ats_copy_rejected_for_its_location_does_not_drop_the_email_copy(env):
    p, ats, mail, scorer, notifier, clock, db = env
    ats.jobs = [gh("1", title="Junior Data Engineer", company="Initech", location="Berlin, Germany")]
    run(p)
    assert status(db, "1")["status"] == "filtered"
    clock.tick()
    mail.jobs = [em("9", title="Junior Data Engineer", company="Initech", location="Cluj-Napoca")]
    run(p)
    assert status(db, "9", "email")["status"] == "digest"


def test_same_role_in_two_emails_is_queued_once(env):
    p, ats, mail, scorer, notifier, clock, db = env
    mail.jobs = [em("1", title="Junior Data Engineer", company="Initech"),
                 em("2", title="Junior Data Engineer", company="Initech", extra={"_from_email": "1", "_email_platform": "indeed"})]
    stats = run(p)
    assert stats.digest_queued == 1 and stats.duplicates == 1


# --- the digest ---------------------------------------------------------------------------------------------------

def at_local(h, m=0):
    return datetime(2026, 11, 8, h, m).astimezone().astimezone(timezone.utc)


def queue(p, mail, clock, *jobs):
    mail.jobs = list(jobs)
    run(p)
    clock.now = at_local(18, 5)


def test_digest_is_sent_once_a_day_after_18_00_sorted_with_hints_and_no_scores(env):
    p, ats, mail, scorer, notifier, clock, db = env
    scorer.scores = {"1": 45, "2": 85, "3": 62}
    p.cfg.notify.digest.title_hint = True
    queue(p, mail, clock, em("1", title="Junior Data Engineer", company="A"), em("2", title="Junior Data Scientist", company="B"),
          em("3", title="Python Intern", company="C"))
    clock.now = at_local(17, 30)
    assert not asyncio.run(digest_mod.maybe_send(p, clock.now))     # not yet
    clock.now = at_local(18, 5)
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    (digest,) = notifier.digests
    assert [(i.job.job_id, i.hint) for i in digest.items] == [("2", "good"), ("3", "maybe"), ("1", "weak")]
    assert notifier.alerts == []                                     # a digest is not an alert
    assert all(status(db, j, "email")["status"] == "digested" for j in "123")
    assert not asyncio.run(digest_mod.maybe_send(p, clock.now + timedelta(minutes=10)))   # once per day
    messages = "\n".join(render_discord_digest(digest))
    assert "/100" not in messages and "score" not in messages.lower().replace("no score", "")


def test_digest_without_hints_by_default_is_unranked_and_never_asks_the_model(env):
    p, ats, mail, scorer, notifier, clock, db = env
    assert p.cfg.notify.digest.title_hint is False
    queue(p, mail, clock, em("1", title="Python Intern", company="B", location="Romania"),
          em("2", title="Junior Data Engineer", company="Z", location="Cluj-Napoca"),
          em("3", title="Junior Data Engineer", company="A", location="Cluj-Napoca"))
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    d = notifier.digests[0]
    assert scorer.calls == [] and d.hint_note is None and all(i.hint is None for i in d.items)
    assert [i.job.job_id for i in d.items] == ["3", "2", "1"]     # alphabetical, Romania-only last
    text = "\n".join(render_discord_digest(d))
    assert "title fit" not in text and "No title-fit" not in text


def test_low_priority_locations_sort_last_and_are_marked(env):
    p, ats, mail, scorer, notifier, clock, db = env
    queue(p, mail, clock, em("1", title="Junior Data Engineer", company="A", location="Romania"),
          em("2", title="Junior Data Scientist", company="B", location="Cluj-Napoca"))
    asyncio.run(digest_mod.maybe_send(p, clock.now))
    items = notifier.digests[0].items
    assert [i.job.job_id for i in items] == ["2", "1"] and items[1].low_priority == "Romania (country only)"
    assert "low priority: Romania (country only)" in "\n".join(render_discord_digest(notifier.digests[0]))


def test_digest_without_hints_when_the_model_cannot_answer(env):
    p, ats, mail, scorer, notifier, clock, db = env
    scorer.unavailable = "Ollama is not reachable"
    p.cfg.notify.digest.title_hint = True
    queue(p, mail, clock, em("1", title="Junior Data Engineer", company="A"))
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    d = notifier.digests[0]
    assert d.items[0].hint is None and "not reachable" in d.hint_note


def test_digest_stays_queued_when_sending_fails_or_paused(env):
    p, ats, mail, scorer, notifier, clock, db = env
    queue(p, mail, clock, em("1", title="Junior Data Engineer", company="A"))
    notifier.fail = True
    assert not asyncio.run(digest_mod.maybe_send(p, clock.now))
    assert status(db, "1", "email")["status"] == "digest"
    notifier.fail = False
    p.control.pause(1)
    assert not asyncio.run(digest_mod.maybe_send(p, clock.now))
    p.control.resume()
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))


def test_digest_max_items_leaves_the_rest_for_tomorrow(env, cfg):
    p, ats, mail, scorer, notifier, clock, db = env
    cfg.notify.digest.max_items = 2
    queue(p, mail, clock, *(em(str(i), title=f"Junior Data Engineer {i}", company=f"C{i}") for i in range(1, 5)))
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    assert len(notifier.digests[0].items) == 2 and notifier.digests[0].queued_more == 2
    assert db.count_status("digest") == 2
    assert not db.get_meta("digest_sent_on")   # today isn't finished: the rest goes out at the next run


def test_discord_digest_messages_fit_and_never_split_an_item():
    long_url = "https://cts.indeed.com/v3/" + "A" * 1500
    items = [DigestItem(em(str(i), title=f"Junior Data Engineer {i}", url=long_url if i % 7 == 0 else f"https://x/{i}"), "good")
             for i in range(60)]
    messages = render_discord_digest(Digest(items=items, day="2026-11-08", queued_more=3))
    assert all(len(m) <= 2000 for m in messages) and len(messages) > 1
    assert "…and 3 more" in messages[-1]
    assert sum(m.count("**") // 2 for m in messages) == 60
    subject, body = render_email_digest(Digest(items=items[:2], day="2026-11-08"))
    assert subject.endswith("2 jobs") and "NOT alerts" in body


def test_console_notifier_prints_the_digest(capsys):
    asyncio.run(ConsoleNotifier().send_digest(Digest(items=[DigestItem(em("1"), None)], day="2026-11-08",
                                                     hint_note="the model was busy")))
    out = capsys.readouterr().out
    assert "DIGEST message 1" in out and "the model was busy" in out


# --- Ollama not answering: start once, hold, then [unscored] -------------------------------------------------------

class DownScorer(FakeScorer):
    unreachable = True

    def __init__(self):
        super().__init__()
        self.unavailable = "Ollama is not reachable at http://localhost:11434 (ConnectError)"


@pytest.fixture
def down_env(cfg, monkeypatch):
    clock = Clock()
    db = DB(cfg.db_path)
    src, scorer, notifier = FakeSource(), DownScorer(), RecordingNotifier()
    launched, waits = [], []

    async def fake_start(client, wait, launcher, poll=2.0):
        waits.append(wait)
        launcher()
        return False, "started the Ollama tray app; still no answer after 90 s"

    monkeypatch.setattr(ollama_ctl, "start_and_wait", fake_start)
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [src], notifier, scorer, clock=clock,
                 ollama_launcher=lambda: launched.append(1) or "the Ollama tray app")
    run(p)
    clock.tick()
    yield p, src, scorer, notifier, clock, db, launched, waits
    db.close()


def test_unreachable_ollama_is_started_once_and_jobs_are_held_then_sent_unscored(down_env, cfg):
    p, src, scorer, notifier, clock, db, launched, waits = down_env
    src.jobs = [gh("1", title="Junior Data Engineer")]
    stats = run(p)
    assert launched == [1] and waits == [cfg.llm.ollama.auto_start_wait_seconds]
    assert stats.notified == 0 and stats.held == 1 and status(db, "1")["status"] == "pending"
    for _ in range(3):                      # later runs of the same outage don't try to start it again
        clock.tick()
        run(p)
    assert launched == [1] and notifier.alerts == []
    clock.tick(minutes=130)                 # past control.defer_max_hold_minutes (120)
    stats = run(p)
    assert stats.notified == 1 and stats.unscored == 1
    assert "scoring was deferred for over 120 min" in notifier.alerts[0].unscored
    assert "not reachable" in notifier.alerts[0].unscored


def test_ollama_that_comes_up_after_the_start_attempt_scores_normally(down_env, monkeypatch):
    p, src, scorer, notifier, clock, db, launched, waits = down_env

    async def start_ok(client, wait, launcher, poll=2.0):
        launcher()
        scorer.unavailable, scorer.unreachable = None, False
        return True, "started the Ollama tray app"

    monkeypatch.setattr(ollama_ctl, "start_and_wait", start_ok)
    src.jobs = [gh("1", title="Junior Data Engineer")]
    stats = run(p)
    assert launched == [1] and stats.scored == 1 and stats.unscored == 0 and notifier.alerts[0].score == 90


def test_a_second_outage_may_start_it_again(down_env):
    p, src, scorer, notifier, clock, db, launched, waits = down_env
    src.jobs = [gh("1", title="Junior Data Engineer")]
    run(p)
    scorer.unavailable, scorer.unreachable = None, False     # it came back (the held job is scored)
    clock.tick()
    run(p)
    scorer.unavailable, scorer.unreachable = "Ollama is not reachable", True
    clock.tick()
    src.jobs.append(gh("2", title="Junior Data Scientist"))
    run(p)
    assert launched == [1, 1]


def test_a_missing_model_is_not_held_it_is_a_configuration_problem(down_env):
    p, src, scorer, notifier, clock, db, launched, waits = down_env
    scorer.unreachable, scorer.unavailable = False, "Ollama model 'x' is not pulled; run `ollama pull x`"
    src.jobs = [gh("1", title="Junior Data Engineer")]
    stats = run(p)
    assert launched == [] and stats.unscored == 1


def test_start_and_wait_waits_for_the_server():
    answers = iter([False, False, True])

    def handler(request):
        if next(answers, False):
            return httpx.Response(200, json={"version": "0"})
        raise httpx.ConnectError("refused")

    async def go(wait):
        async with httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler)) as c:
            return await ollama_ctl.start_and_wait(c, wait, lambda: "the Ollama tray app", poll=0.01)

    ok, note = asyncio.run(go(5))
    assert ok and "tray app" in note
    ok, note = asyncio.run(go(0.05))   # never answers within the wait (the iterator is exhausted: handler raises)
    assert not ok and "still no answer" in note


def test_start_and_wait_survives_a_launcher_that_fails():
    def boom():
        raise OSError("no explorer")

    async def go():
        async with httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(lambda r: httpx.Response(200))) as c:
            return await ollama_ctl.start_and_wait(c, 1, boom, poll=0.01)

    ok, note = asyncio.run(go())
    assert not ok and "could not start Ollama" in note


def test_a_scorer_going_down_mid_run_holds_the_rest(cfg):
    clock, db, src, notifier = Clock(), DB(cfg.db_path), FakeSource(), RecordingNotifier()
    scorer = FakeScorer()
    scorer.fail = ScoreUnreachable("Ollama is not reachable at x (ConnectError)")
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [src], notifier, scorer, clock=clock)
    run(p)
    clock.tick()
    src.jobs = [gh("1", title="Junior Data Engineer"), gh("2", title="Junior Data Scientist")]
    stats = run(p)
    assert stats.notified == 0 and stats.held == 2
    assert {status(db, "1")["status"], status(db, "2")["status"]} == {"pending"}
    db.close()


# --- latency: bot offline ------------------------------------------------------------------------------------------

def iso(h, m=0, day=8):
    return datetime(2026, 11, day, h, m, tzinfo=timezone.utc).isoformat()


def runs_every_10_min(start_h, end_h, day=8):
    t, out = datetime(2026, 11, day, start_h, 0, tzinfo=timezone.utc), []
    while t < datetime(2026, 11, day, end_h, 0, tzinfo=timezone.utc):
        out.append(t)
        t += timedelta(minutes=10)
    return out


def test_bot_was_offline_rule():
    runs = runs_every_10_min(8, 10) + runs_every_10_min(14, 16)     # the bot was off from 10:00 to 14:00
    # posted 10:30 and first seen 14:00: no run in [10:30, 13:50]  -> offline
    assert bot_was_offline(iso(10, 30), iso(14, 0), runs, 10)
    # an ordinary delay with runs throughout
    assert not bot_was_offline(iso(8, 5), iso(8, 40), runs, 10)
    # detected by the first run after posting: the window is empty, not offline
    assert not bot_was_offline(iso(8, 3), iso(8, 10), runs, 10)
    # posted just after a run, found by the next one: window shorter than the slack, not offline
    assert not bot_was_offline(iso(8, 1), iso(8, 20), runs, 10)
    # nothing is known about jobs first seen before the oldest stored run, or without posted_at
    assert not bot_was_offline(iso(1, 0), iso(5, 0), runs, 10)
    assert not bot_was_offline(None, iso(14, 0), runs, 10)
    assert not bot_was_offline(iso(10, 30), iso(14, 0), [], 10)    # no runs recorded at all


def test_report_leaves_offline_jobs_out_of_the_percentiles(cfg):
    db = DB(cfg.db_path)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    starts = [now - timedelta(hours=10) + timedelta(minutes=10 * i) for i in range(0, 20)] + \
             [now - timedelta(hours=2) + timedelta(minutes=10 * i) for i in range(0, 12)]   # offline for ~7 hours
    for s in starts:
        db.record_run(s, s + timedelta(seconds=20), {})
    def add(job_id, posted, seen):
        j = make_job(job_id, posted_at=posted)
        db.insert_job(j, status="new", first_seen_at=seen, crosspost_key=job_id)
    add("fast", starts[2] + timedelta(minutes=1), starts[3])               # 9 min
    add("slow-but-online", starts[2] + timedelta(minutes=1), starts[2] + timedelta(minutes=50))
    add("downtime", starts[19] + timedelta(minutes=1), starts[20])         # posted in the gap, seen at restart
    text = build_report(db, interval_minutes=10)
    line = next(ln for ln in text.splitlines() if ln.startswith("fake"))
    cols = line.split()
    assert cols[1] == "2" and cols[3] == "1"          # n = 2, bot offline = 1
    assert "bot offline" in text
    db.close()


# --- digest.require_entry_level_title (email jobs only) -------------------------------------------------------------

def test_email_job_needs_an_entry_level_word_in_its_title(env):
    p, ats, mail, scorer, notifier, clock, db = env
    with_entry_keywords(p)
    mail.jobs = [em("1", title="Python Developer"), em("2", title="Junior Python Developer"),
                 em("3", title="Data Engineer Intern"), em("4", title="Software Engineer (Graduate Programme)"),
                 em("5", title="Stagiar Data Engineer")]
    ats.jobs = [gh("9", title="Python Developer", company="Other")]   # ATS jobs are unchanged: no entry-level word needed
    run(p)
    assert status(db, "1", "email")["status"] == "filtered"
    assert status(db, "1", "email")["reason"] == "email: no entry-level signal"
    assert {status(db, j, "email")["status"] for j in "2345"} == {"digest"}
    assert status(db, "9")["status"] == "notified"


def with_entry_keywords(p):
    """The shared test config has no profile.entry_level_keywords (the rule is inactive without them)."""
    p.cfg.profile.entry_level_keywords = ["junior", "intern", "graduate", "stagiar"]
    p.rules = RuleFilter(p.cfg)


def test_entry_level_requirement_can_be_switched_off(cfg):
    cfg.profile.entry_level_keywords = ["junior"]
    cfg.notify.digest.require_entry_level_title = False
    rules = RuleFilter(cfg)
    assert rules.prefilter(em("1", title="Python Developer", location="Cluj-Napoca")).ok


def test_queued_email_job_is_rechecked_when_the_digest_is_built(env):
    p, ats, mail, scorer, notifier, clock, db = env
    p.cfg.notify.digest.require_entry_level_title = False
    with_entry_keywords(p)
    queue(p, mail, clock, em("1", title="Python Developer", company="A"), em("2", title="Junior Python Developer", company="B"))
    assert status(db, "1", "email")["status"] == "digest"
    p.cfg.notify.digest.require_entry_level_title = True      # turned on while it was waiting
    p.rules = RuleFilter(p.cfg)
    assert asyncio.run(digest_mod.maybe_send(p, clock.now))
    assert [i.job.job_id for i in notifier.digests[0].items] == ["2"]
    assert status(db, "1", "email")["reason"] == "email: no entry-level signal"
