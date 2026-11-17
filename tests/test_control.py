"""Pause / defer controls: state transitions, the pending-job hold, auto-defer on low VRAM, unloading the model."""
import asyncio
import json
import sys
from datetime import timedelta

import httpx
import pytest

from jobbot import gpu, ollama_ctl
from jobbot.control import Control, ControlState
from jobbot.db import DB
from jobbot.heartbeat import build_message
from jobbot.pipeline import Pipeline

from .conftest import mock_client
from .gpu_fakes import Rig, make_guard
from .test_pipeline import Clock, FakeScorer, FakeSource, RecordingNotifier, gh, run, status


@pytest.fixture
def env(cfg):
    clock = Clock()
    db = DB(cfg.db_path)
    src, matcher, notifier = FakeSource(), FakeScorer(), RecordingNotifier()
    rig = Rig()
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [src], notifier, matcher, clock=clock,
                 guard=make_guard(cfg, db, clock, rig))
    src.jobs = [gh("1")]
    run(p)  # seeds job 1
    clock.tick()
    yield p, src, matcher, notifier, clock, db, rig
    db.close()


TITLES = {"2": "Junior Data Scientist", "3": "Junior Python Developer", "4": "Junior ETL Developer"}


def add_job(src, job_id="2"):
    """Distinct titles, or cross-post dedup would merge the jobs."""
    src.jobs.append(gh(job_id, title=TITLES[job_id]))


# --- state transitions -----------------------------------------------------

def test_pause_auto_resumes_after_the_configured_hours(env):
    p, *_, clock, db, _ = env
    c = p.control
    c.pause()
    assert c.refresh().paused
    clock.tick(11 * 60)
    assert c.refresh().paused
    clock.tick(61)  # 12 h 1 min after the pause (the fixture's clock started 10 min earlier)
    assert not c.refresh().paused and c.read().mode == "running"
    # The pause period ends when the timer ran out, not when it was noticed.
    assert 11.9 * 3600 < db.period_seconds("paused", clock.now - timedelta(days=1), clock.now) < 12.1 * 3600


def test_pause_hours_zero_means_until_resume_and_resume_clears_everything(env):
    p, *_, clock, db, _ = env
    c = p.control
    c.pause(0)
    c.defer("resume")
    clock.tick(48 * 60)
    assert c.refresh().paused and c.read().deferred
    before, after = c.resume()
    assert before.mode == "paused" and after.mode == "running" and not after.deferred


def test_timed_defer_expires_and_next_scan_defer_ends_when_a_run_starts(env):
    p, *_, clock, db, _ = env
    c = p.control
    c.defer("minutes", 30)
    clock.tick(29)
    assert c.refresh().deferred
    clock.tick(2)
    assert not c.refresh().deferred
    c.defer("next_scan")
    assert c.refresh().deferred and c.scoring_block() is not None  # still deferred until the next run begins
    assert not c.begin_run().deferred


def test_state_survives_a_restart(env):
    p, *_, clock, db, _ = env
    p.control.pause(2)
    p.control.defer("minutes", 90)
    db2 = DB(db.conn.execute("PRAGMA database_list").fetchone()["file"])  # a second process / the bot after a reboot
    try:
        s = Control(db2, p.cfg, clock).read()
        assert s.paused and s.deferred and s.defer_kind == "minutes"
        assert s.paused_until == clock.now + timedelta(hours=2)
    finally:
        db2.close()


def test_garbage_state_is_ignored(env):
    p, *_, db, _ = env
    db.set_meta("control", "{nope")
    assert p.control.read() == ControlState()


# --- the pipeline ----------------------------------------------------------

def test_paused_run_does_nothing_and_is_not_recorded(env):
    p, src, matcher, notifier, clock, db, _ = env
    add_job(src)
    runs_before = len(db.runs_since(clock.now - timedelta(days=1)))
    p.control.pause()
    stats = run(p)
    assert stats.paused == 1 and stats.fetched == 0
    assert db.job_exists(("fake", "2")) is False  # nothing was fetched or stored
    assert matcher.calls == [] and notifier.alerts == []
    assert len(db.runs_since(clock.now - timedelta(days=1))) == runs_before
    # After the resume the next run catches up: the job posted during the pause is simply new.
    p.control.resume()
    clock.tick()
    run(p)
    assert status(db, "2")["status"] == "notified"


def test_no_feed_warnings_while_paused(env):
    p, *_, notifier, clock, db, _ = env[:1] + env[1:]
    p.control.pause()
    assert asyncio.run(p._notify_text("feed down")) is False
    assert notifier.texts == []
    p.control.resume()
    assert asyncio.run(p._notify_text("feed down")) is True


def test_deferred_jobs_wait_as_pending_then_are_scored_when_deferral_ends(env):
    p, src, matcher, notifier, clock, db, _ = env
    add_job(src)
    p.control.defer("minutes", 30)
    stats = run(p)
    assert stats.held == 1 and stats.passed_rules == 1 and stats.scored == 0
    assert status(db, "2")["status"] == "pending" and matcher.calls == [] and notifier.alerts == []
    # Still fetching and filtering during the deferral: a second job arrives and waits too.
    add_job(src, "3")
    clock.tick(5)
    run(p)
    assert db.count_status("pending") == 2
    clock.tick(30)  # deferral over; the bot's idle loop calls drain_pending
    assert not p.control.refresh().deferred
    stats = asyncio.run(p.drain_pending())
    assert stats.scored == 2 and stats.passed_rules == 0  # waiting jobs aren't counted as passing the rules twice
    assert db.count_status("pending") == 0 and len(notifier.alerts) == 2
    assert all(a.score == 90 and a.unscored is None for a in notifier.alerts)


def test_pending_job_goes_out_unscored_after_the_hold_limit(env):
    p, src, matcher, notifier, clock, db, _ = env
    add_job(src)
    p.control.defer("resume")
    run(p)
    assert status(db, "2")["status"] == "pending"
    clock.tick(110)
    run(p)
    assert status(db, "2")["status"] == "pending"  # 110 min < the 120 min default
    clock.tick(15)
    run(p)
    row = status(db, "2")
    assert row["status"] == "notified" and "unscored" in row["reason"]
    assert notifier.alerts[0].score is None and "deferred" in notifier.alerts[0].unscored
    assert matcher.calls == []


def test_hold_limit_null_waits_indefinitely(cfg, env):
    p, src, matcher, notifier, clock, db, _ = env
    p.cfg.control.defer_max_hold_minutes = None
    add_job(src)
    p.control.defer("resume")
    run(p)
    clock.tick(24 * 60)
    run(p)
    assert status(db, "2")["status"] == "pending" and notifier.alerts == []


def test_pause_during_a_run_leaves_the_rest_pending(env):
    p, src, matcher, notifier, clock, db, _ = env
    add_job(src)
    add_job(src, "3")
    original = matcher.score

    async def pause_after_first(job, note=None, *, second_opinion=False):
        r = await original(job, note, second_opinion=second_opinion)
        p.control.pause()
        return r

    matcher.score = pause_after_first
    run(p)
    assert len(matcher.calls) == 1
    assert sorted(r["status"] for r in (status(db, "2"), status(db, "3"))) == ["notified", "pending"]


def test_a_pending_job_that_no_longer_passes_the_rules_is_filtered(env):
    p, src, matcher, notifier, clock, db, _ = env
    add_job(src)
    p.control.defer("resume")
    run(p)
    db.update_job(("fake", "2"), title="Senior Data Engineer")
    p.control.resume()
    asyncio.run(p.drain_pending())
    assert status(db, "2")["status"] == "filtered" and matcher.calls == []


# --- the GPU guard in the pipeline (each trigger is covered in test_guard.py) ------------------

def test_a_busy_gpu_defers_scoring_and_the_next_scan_scores_once_it_is_free(env):
    p, src, matcher, notifier, clock, db, rig = env
    matcher.uses_gpu = True
    rig.models = ["ministral-3:8b"]  # another project's model
    add_job(src)
    stats = run(p)
    assert stats.auto_deferred == 1 and stats.held == 1 and matcher.calls == []
    assert status(db, "2")["status"] == "pending" and "ministral-3:8b" in status(db, "2")["reason"]
    rig.models = []
    clock.tick()
    stats = run(p)
    assert stats.scored == 1 and status(db, "2")["status"] == "notified"


def test_a_hosted_backend_never_waits_for_the_gpu(env):
    p, src, matcher, notifier, clock, db, rig = env
    rig.models = ["ministral-3:8b"]
    rig.util = 99  # matcher.uses_gpu is False
    add_job(src)
    assert run(p).scored == 1


def test_unreadable_gpu_does_not_defer(env):
    p, src, matcher, notifier, clock, db, rig = env
    matcher.uses_gpu = True
    rig.smi_ok = rig.ollama_ok = False
    add_job(src)
    assert run(p).scored == 1


def test_auto_defer_can_be_turned_off_but_an_explicit_flag_still_counts(env):
    p, src, matcher, notifier, clock, db, rig = env
    p.cfg.control.auto_defer = False
    matcher.uses_gpu = True
    rig.util, rig.free_mb, rig.models = 95, 100, ["ministral-3:8b"]
    add_job(src)
    assert run(p).scored == 1
    from jobbot.guard import set_flag
    set_flag(db, "benchmark", None, clock.now)
    add_job(src, "3")
    clock.tick()
    assert run(p).held == 1


# --- nvidia-smi and Ollama stand-ins --------------------------------------------

def fake_smi(tmp_path, body):
    script = tmp_path / "fake_smi.py"
    script.write_text(body, encoding="utf-8")
    return (sys.executable, str(script))


def test_free_vram_reads_nvidia_smi_output(tmp_path):
    assert asyncio.run(gpu.free_vram_mb(fake_smi(tmp_path, "print('3500')"))) == 3500
    assert asyncio.run(gpu.free_vram_mb(fake_smi(tmp_path, "print('900'); print('11800')"))) == 11800  # best GPU
    assert asyncio.run(gpu.free_vram_mb(fake_smi(tmp_path, "import sys; sys.exit(9)"))) is None
    assert asyncio.run(gpu.free_vram_mb(fake_smi(tmp_path, "print('N/A')"))) is None
    assert asyncio.run(gpu.free_vram_mb(("definitely-not-nvidia-smi",))) is None


class FakeOllamaServer:
    def __init__(self, loaded=True, unloads=True):
        self.loaded, self.unloads, self.generate_calls = loaded, unloads, []

    def __call__(self, request):
        if request.url.path == "/api/ps":
            models = [{"name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 6, "size_vram": 6,
                       "expires_at": "2026-10-30T22:00:00Z"}] if self.loaded else []
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/generate":
            body = json.loads(request.content)
            self.generate_calls.append(body)
            if body["keep_alive"] == 0 and self.unloads:
                self.loaded = False
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)


def client_for(fake):
    return httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(fake))


def test_unload_sends_keep_alive_zero_and_waits_until_gone():
    fake = FakeOllamaServer()

    async def go():
        async with client_for(fake) as c:
            return await ollama_ctl.unload(c, "qwen3.5:9b")
    assert asyncio.run(go()) is True
    assert fake.generate_calls == [{"model": "qwen3.5:9b", "keep_alive": 0}]


def test_unload_does_not_touch_a_model_that_is_not_loaded():
    fake = FakeOllamaServer(loaded=False)

    async def go():
        async with client_for(fake) as c:
            return await ollama_ctl.unload(c, "qwen3.5:9b")
    assert asyncio.run(go()) is True and fake.generate_calls == []


def test_unload_reports_failure_when_the_model_stays_loaded():
    fake = FakeOllamaServer(unloads=False)

    async def go():
        async with client_for(fake) as c:
            return await ollama_ctl.unload(c, "qwen3.5:9b", wait=0.6)
    assert asyncio.run(go()) is False


# --- heartbeat ------------------------------------------------------------------

def test_heartbeat_reports_time_paused_and_deferred(env):
    p, src, matcher, notifier, clock, db, _ = env
    p.control.pause(3)
    clock.tick(90)
    p.control.resume()
    p.control.defer("minutes", 45)
    clock.tick(50)
    p.control.refresh()
    text = build_message(db, clock.now)
    assert "paused 1 h 30 min" in text and "scoring deferred 45 min" in text
