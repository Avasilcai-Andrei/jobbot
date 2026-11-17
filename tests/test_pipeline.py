import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from jobbot.db import DB
from jobbot.scoring import MatchResult, ScoreFailed, ScoreUnavailable
from jobbot.scoring.base import Scorer
from jobbot.pipeline import Pipeline
from jobbot.report import build_report
from jobbot.sources.base import Source

from .conftest import make_job, mock_client


class FakeSource(Source):
    kind = "fake"

    def __init__(self, name="acme", company="Acme"):
        super().__init__(name, company)
        self.jobs = []
        self.fail = False
        self.detail_calls = 0

    @classmethod
    def from_entry(cls, entry):
        return cls(entry["name"])

    async def fetch(self, http):
        if self.fail:
            raise httpx.ConnectError("boom")
        return [j for j in self.jobs]

    async def fetch_details(self, http, job):
        self.detail_calls += 1
        job.description = job.extra.get("_desc", "Python, SQL. Entry level.")
        return job


class FakeScorer:
    name = "fake:model"
    uses_gpu = False

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []
        self.fail = None          # an exception instance to raise from score()
        self.unavailable = None   # a reason returned by check_available()
        self.checks = 0
        self.loaded = False
        self.unloads = 0

    async def check_available(self):
        self.checks += 1
        return self.unavailable

    async def model_loaded(self):
        return self.loaded

    async def unload(self):
        self.unloads += 1
        self.loaded = False

    async def score(self, job, note=None, *, second_opinion=False):
        self.calls.append(job.job_id + ("/2" if second_opinion else ""))
        if self.fail:
            raise self.fail
        s = self.scores.get(job.job_id, 90)  # an int, or (first, second opinion)
        s = (s[1] if second_opinion else s[0]) if isinstance(s, tuple) else s
        return MatchResult(score=s, reason="second" if second_opinion else "fits", missing_requirements=[])

    score_stable = Scorer.score_stable  # the real near-threshold logic, on top of the fake score()


class RecordingNotifier:
    def __init__(self):
        self.alerts = []
        self.texts = []
        self.fail = False

    async def send_alert(self, alert):
        if self.fail:
            raise RuntimeError("discord down")
        self.alerts.append(alert)

    async def send_text(self, text):
        self.texts.append(text)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def tick(self, minutes=10):
        self.now += timedelta(minutes=minutes)


def gh(job_id, **kw):
    """A job as a list endpoint returns it: no description yet."""
    kw.setdefault("description", None)
    kw.setdefault("feed", "fake:acme")
    return make_job(job_id, **kw)


@pytest.fixture
def env(cfg):
    clock = Clock()
    db = DB(cfg.db_path)
    src = FakeSource()
    matcher = FakeScorer()
    notifier = RecordingNotifier()
    http = mock_client(lambda r: httpx.Response(500))
    p = Pipeline(cfg, db, http, [src], notifier, matcher, clock=clock)
    yield p, src, matcher, notifier, clock, db
    db.close()


def run(p):
    return asyncio.run(p.run_once())


def status(db, job_id, source="fake"):
    return db.conn.execute("SELECT * FROM jobs WHERE source=? AND job_id=?", (source, job_id)).fetchone()


def test_first_run_seeds_without_notifying(env):
    p, src, matcher, notifier, clock, db = env
    src.jobs = [gh("1"), gh("2", title="Junior Python Developer")]
    stats = run(p)
    assert stats.seeded == 2 and stats.new == 0
    assert notifier.alerts == [] and matcher.calls == [] and src.detail_calls == 0
    assert status(db, "1")["status"] == "seeded"


def test_new_job_flows_to_notification(env):
    p, src, matcher, notifier, clock, db = env
    src.jobs = [gh("1")]
    run(p)
    clock.tick()
    posted = clock.now - timedelta(minutes=4)
    src.jobs.append(gh("2", title="Junior Data Engineer II", posted_at=posted))
    stats = run(p)
    assert stats.new == 1 and stats.notified == 1
    assert matcher.calls == ["2"] and src.detail_calls == 1
    row = status(db, "2")
    assert row["status"] == "notified" and row["score"] == 90 and row["notified_at"]
    assert row["description"]  # stored for retries / cache
    assert notifier.alerts[0].first_seen_at == clock.now
    # seen again next run: nothing happens
    clock.tick()
    assert run(p).new == 0 and len(notifier.alerts) == 1


def test_rules_and_threshold(env):
    p, src, matcher, notifier, clock, db = env
    run(p)  # seed an empty feed
    clock.tick()
    src.jobs = [
        gh("10", title="Senior Data Engineer"),
        gh("11", title="Junior Data Engineer", location="Berlin, Germany"),
        gh("12", title="Data Engineer", extra={"_desc": "You need 5+ years of experience."}),
        gh("13", title="Junior Data Scientist"),
        gh("14", title="Junior Python Developer", location="Bucharest, Romania"),
    ]
    matcher.scores = {"13": 40}
    stats = run(p)
    assert stats.filtered == 3 and stats.scored_low == 1 and stats.notified == 1
    assert src.detail_calls == 3  # no description fetch for title/location rejects
    assert status(db, "12")["reason"] == "asks for 5+ years of experience"
    assert status(db, "13")["status"] == "scored"
    assert notifier.alerts[0].job.job_id == "14" and notifier.alerts[0].low_priority


def test_crosspost_prefers_high_priority_location(env, cfg):
    p, src, matcher, notifier, clock, db = env
    other = FakeSource("acme-careers")
    p.sources.append(other)
    p.by_feed[other.feed] = other
    run(p)
    clock.tick()
    src.jobs = [gh("20", title="Junior Data Engineer", location="Bucharest, Romania")]
    other.jobs = [gh("21", title="Junior Data Engineer (m/f/d)", location="Cluj-Napoca", feed=other.feed)]
    stats = run(p)
    assert stats.notified == 1 and stats.duplicates == 1
    assert notifier.alerts[0].job.job_id == "21"  # the Cluj posting wins
    assert status(db, "20")["status"] == "duplicate"
    # a repost much later is not a duplicate any more
    clock.tick(60 * 24 * 40)
    src.jobs.append(gh("22", title="Junior Data Engineer", location="Cluj"))
    assert run(p).notified == 1


def test_scoring_failure_sends_unscored_instead_of_dropping(env):
    p, src, matcher, notifier, clock, db = env
    run(p)
    clock.tick()
    src.jobs = [gh("30")]
    matcher.fail = ScoreFailed("malformed output twice")
    stats = run(p)
    assert stats.notified == 1 and stats.unscored == 1
    a = notifier.alerts[0]
    assert a.score is None and "malformed" in a.unscored
    row = status(db, "30")
    assert row["status"] == "notified" and row["reason"].startswith("unscored:")


def test_backend_down_checked_once_and_everything_goes_unscored(env, caplog):
    p, src, matcher, notifier, clock, db = env
    run(p)
    clock.tick()
    src.jobs = [gh("31"), gh("32", title="Junior Python Developer")]
    matcher.unavailable = "Ollama is not reachable at http://localhost:11434"
    with caplog.at_level("WARNING"):
        stats = run(p)
    assert stats.unscored == 2 and matcher.calls == [] and matcher.checks == 1
    warnings = [r for r in caplog.records if "Scoring unavailable" in r.getMessage()]
    assert len(warnings) == 1  # one clear warning per run, not per job


def test_backend_dies_mid_run(env):
    p, src, matcher, notifier, clock, db = env
    run(p)
    clock.tick()
    src.jobs = [gh("33"), gh("34", title="Junior Python Developer")]
    matcher.fail = ScoreUnavailable("connection refused")
    stats = run(p)
    assert stats.notified == 2 and stats.unscored == 2


def test_backend_off(env):
    p, src, matcher, notifier, clock, db = env
    p.scorer, p.scorer_off_reason = None, "scoring is off (llm.backend: off)"
    run(p)
    clock.tick()
    src.jobs = [gh("35")]
    run(p)
    assert notifier.alerts[0].unscored == "scoring is off (llm.backend: off)"


def test_notify_failure_retries(env):
    p, src, matcher, notifier, clock, db = env
    run(p)
    clock.tick()
    src.jobs = [gh("40")]
    notifier.fail = True
    run(p)
    assert status(db, "40")["status"] == "new"
    notifier.fail = False
    clock.tick()
    run(p)
    assert status(db, "40")["status"] == "notified"


def test_health_warning_after_three_failures_and_recovery(env):
    p, src, matcher, notifier, clock, db = env
    run(p)
    src.fail = True
    for _ in range(2):
        clock.tick()
        run(p)
    assert notifier.texts == []
    clock.tick()
    run(p)
    assert len(notifier.texts) == 1 and "fake:acme" in notifier.texts[0] and "3 runs" in notifier.texts[0]
    # 4th failure: no repeated warning; then backoff skips a run
    clock.tick()
    run(p)
    clock.tick()
    assert run(p).feeds_skipped == 1
    assert len(notifier.texts) == 1
    src.fail = False
    clock.tick(20)
    run(p)
    assert len(notifier.texts) == 2 and "working again" in notifier.texts[1]
    assert db.feed_state("fake:acme").consecutive_failures == 0


def test_report_shows_latency(env):
    p, src, matcher, notifier, clock, db = env
    run(p)
    for i, lag in enumerate([3, 8, 12, 45]):
        clock.tick()
        src.jobs.append(gh(f"5{i}", title=f"Junior Data Engineer {i}", posted_at=clock.now - timedelta(minutes=lag)))
        run(p)
    text = build_report(db)
    line = next(ln for ln in text.splitlines() if ln.startswith("fake"))
    assert line.split()[1] == "4"  # n
    assert "75%" in line  # 3 of 4 within 30 minutes


def test_near_threshold_scores_twice_and_uses_the_mean(env):
    p, src, matcher, notifier, clock, db = env
    run(p)  # seed an empty feed
    clock.tick()
    src.jobs = [gh("20", title="Junior Data Engineer"), gh("21", title="Junior Data Scientist"),
                gh("22", title="Junior Python Developer"), gh("23", title="Data Engineer Intern")]
    # threshold 65, margin 15: 60 -> second opinion 72 -> mean 66 (alert); 70 -> 56 -> 63 (no alert);
    # 90 and 49 are more than 15 away, so they are scored once
    matcher.scores = {"20": (60, 72), "21": (70, 56), "22": 90, "23": 49}
    stats = run(p)
    assert sorted(matcher.calls) == ["20", "20/2", "21", "21/2", "22", "23"]
    assert stats.rescored == 2 and stats.notified == 2 and stats.scored_low == 2
    alert = next(a for a in notifier.alerts if a.job.job_id == "20")
    assert alert.score == 66 and "Scored twice near the threshold: 60 and 72." in alert.reason
    assert alert.reason.startswith("second")  # the explanation that agrees with the decision is kept
    row = status(db, "21")
    assert row["score"] == 63 and row["reason"] == "score 63 < 65 (scored twice: 70 and 56)"
    assert json.loads(row["match_json"])["scores"] == [70, 56]
    assert "2 of 4 scored jobs (50%); 1 of those were alerted" in build_report(db)


def test_rescore_margin_off(env, cfg):
    p, src, matcher, notifier, clock, db = env
    cfg.llm.rescore_margin = None
    run(p)
    clock.tick()
    src.jobs = [gh("30", title="Junior Data Engineer")]
    matcher.scores = {"30": (64, 90)}
    assert run(p).rescored == 0 and matcher.calls == ["30"]


def test_each_run_is_recorded_for_the_heartbeat(env):
    p, src, matcher, notifier, clock, db = env
    run(p)  # seed
    clock.tick()
    src.jobs = [gh("40", title="Junior Data Engineer"), gh("41", title="Senior Data Engineer")]
    matcher.scores = {"40": 90}
    run(p)
    runs = db.runs_since(clock.now - timedelta(hours=1))
    assert len(runs) == 2
    stats = runs[-1][2]
    assert stats["fetched"] == 2 and stats["new"] == 2 and stats["passed_rules"] == 1
    assert stats["scored"] == 1 and stats["notified"] == 1 and stats["filtered"] == 1


class OtherSource(FakeSource):
    kind = "other"


def maintenance_error():
    from jobbot.http import BadResponse

    req = httpx.Request("POST", "https://wk.wd3.myworkdayjobs.com/wday/cxs/wk/External/jobs")
    resp = httpx.Response(200, headers={"content-type": "text/html"}, text="<html>Maintenance</html>", request=req)
    return BadResponse(resp, "not JSON (Expecting value: line 1 column 1 (char 0))")


def test_feeds_of_one_kind_failing_together_get_one_grouped_warning_and_recovery(cfg):
    clock, db, notifier = Clock(), DB(cfg.db_path), RecordingNotifier()
    fakes = [FakeSource(n, n.title()) for n in ("acme", "beta", "gamma")]
    other = OtherSource("solo", "Solo")
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [*fakes, other], notifier, FakeScorer(),
                 clock=clock)

    async def maintenance(http):
        raise maintenance_error()

    run(p)  # seed
    for s in fakes:
        s.fetch = maintenance
    other.fail = True
    for _ in range(3):
        clock.tick()
        run(p)
    assert len(notifier.texts) == 2
    grouped, single = notifier.texts
    assert grouped.startswith("⚠️ jobbot: 3 fake sources have failed 3 runs in a row: `acme`, `beta`, `gamma`.")
    assert "All with the same kind of error (HTTP 200 text/html (not JSON))." in grouped
    assert "Last error (`fake:acme`): BadResponse: not JSON" in grouped and "<html>Maintenance</html>" in grouped
    assert single.startswith("⚠️ jobbot: source `other:solo` has failed 3 runs in a row.")
    assert all(db.feed_state(s.feed).warned for s in [*fakes, other])

    # Two recover while the third keeps failing: one grouped recovery, no new warning for gamma.
    del fakes[0].fetch, fakes[1].fetch
    clock.tick(60)
    run(p)
    assert notifier.texts[2:] == ["✅ jobbot: 2 fake sources are working again: `acme`, `beta`."]
    del fakes[2].fetch
    other.fail = False
    clock.tick(120)
    run(p)
    assert notifier.texts[3:] == ["✅ jobbot: source `fake:gamma` is working again.",
                                  "✅ jobbot: source `other:solo` is working again."]
    db.close()


def test_grouped_warning_is_retried_if_it_could_not_be_sent(cfg):
    clock, db, notifier = Clock(), DB(cfg.db_path), RecordingNotifier()
    fakes = [FakeSource(n, n.title()) for n in ("acme", "beta")]
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), fakes, notifier, FakeScorer(), clock=clock)
    run(p)
    for s in fakes:
        s.fail = True

    async def broken(text):
        raise RuntimeError("discord down")
    notifier.send_text = broken
    for _ in range(3):
        clock.tick()
        run(p)
    assert not any(db.feed_state(s.feed).warned for s in fakes)
    del notifier.send_text
    clock.tick(20)
    run(p)
    assert len(notifier.texts) == 1 and notifier.texts[0].startswith("⚠️ jobbot: 2 fake sources have failed 4 runs")
    assert "(ConnectError)" in notifier.texts[0]
    db.close()
