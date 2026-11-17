"""GPU guard: every "busy" trigger, the quiet window, own-scoring exclusion, stale markers, per-reason time."""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import timedelta

import httpx
import pytest
import yaml

from jobbot import cli, gpu, ollama_ctl
from jobbot.db import DB, utcnow
from jobbot.guard import clear_flag, read_flag, read_lock, set_flag
from jobbot.heartbeat import build_message
from jobbot.pipeline import Pipeline

from .conftest import make_config, mock_client
from .gpu_fakes import OWN, Rig, make_guard
from .test_control import FakeOllamaServer, add_job, client_for
from .test_pipeline import Clock, FakeScorer, FakeSource, RecordingNotifier, gh, run, status


@pytest.fixture
def g(tmp_path):
    cfg = make_config(tmp_path)
    cfg.control.busy_process_patterns = ["gpu.bench", "RENDERJOB"]
    clock, db, rig = Clock(), DB(cfg.db_path), Rig()
    yield cfg, db, clock, rig, make_guard(cfg, db, clock, rig)
    db.close()


def check(guard):
    return asyncio.run(guard.check())


def kinds(a):
    return [r.kind for r in a.reasons]


def lock_path(cfg):
    p = cfg.resolve(type(cfg.base_dir)(cfg.control.busy_lock_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --- other Ollama models --------------------------------------------------------------

def test_another_ollama_model_loaded_is_busy_but_our_own_is_not(g):
    cfg, db, clock, rig, guard = g
    assert not check(guard).busy
    rig.models = [OWN]
    assert not check(guard).busy
    rig.models = [f"{OWN}:latest"]
    assert not check(guard).busy
    rig.models = [OWN, "ministral-3:8b"]
    a = check(guard)
    assert kinds(a) == ["other_model"] and "ministral-3:8b" in a.reasons[0].text and OWN not in a.reasons[0].text


def test_unreachable_ollama_is_not_busy(g):
    cfg, db, clock, rig, guard = g
    rig.ollama_ok = False
    assert not check(guard).busy


# --- utilization and the quiet window -------------------------------------------------------

def test_utilization_above_the_limit_holds_scoring_until_the_window_is_quiet(g):
    cfg, db, clock, rig, guard = g
    rig.util = 20  # the limit is "above 20"
    assert not check(guard).busy
    rig.util = 63
    a = check(guard)
    assert kinds(a) == ["util"] and "63%" in a.reasons[0].text and a.allowed_at == clock.now + timedelta(minutes=10)
    rig.util = 3
    for minutes in (1, 5, 8):
        clock.now = a.at + timedelta(minutes=minutes)
        assert kinds(check(guard)) == ["util"], minutes   # still inside the quiet window
    clock.now = a.at + timedelta(minutes=10, seconds=1)
    quiet = check(guard)
    assert not quiet.busy and quiet.quiet_minutes == 10


def test_a_busy_sample_restarts_the_quiet_window(g):
    cfg, db, clock, rig, guard = g
    rig.util = 50
    first = check(guard)
    clock.now += timedelta(minutes=8)
    check(guard)  # still busy, window from the first sample
    clock.now += timedelta(minutes=1)
    second = check(guard)  # utilization is still 50%: the window now ends 10 min after this sample
    assert second.allowed_at == clock.now + timedelta(minutes=10) and second.allowed_at > first.allowed_at


def test_quiet_time_is_reported_from_what_the_sampler_has_seen(g):
    cfg, db, clock, rig, guard = g
    assert check(guard).quiet_minutes == 0
    clock.now += timedelta(minutes=4)
    assert check(guard).quiet_minutes == 4


def test_our_own_scoring_does_not_count_as_someone_elses_gpu_use(g):
    cfg, db, clock, rig, guard = g
    rig.util = 97
    with guard.scoring():
        assert not check(guard).busy       # sampled while we score
    clock.now += timedelta(seconds=3)
    assert not check(guard).busy           # just after: still our own work
    clock.now += timedelta(seconds=10)
    assert kinds(check(guard)) == ["util"]  # now the same utilization is somebody else's


def test_samples_older_than_fifteen_minutes_are_dropped(g):
    cfg, db, clock, rig, guard = g
    for _ in range(40):
        check(guard)
        clock.now += timedelta(seconds=30)
    assert len(guard.samples) <= 31 and guard.samples[-1].at - guard.samples[0].at <= timedelta(minutes=15)


# --- VRAM and processes ----------------------------------------------------------------

def test_low_free_vram_is_busy_unless_our_model_is_already_loaded(g):
    cfg, db, clock, rig, guard = g
    rig.free_mb = 3_500
    a = check(guard)
    assert kinds(a) == ["vram"] and "3.4 GB" in a.reasons[0].text and "need 8 GB" in a.reasons[0].text
    rig.models = [OWN]  # our own model is what fills the GPU
    assert not check(guard).busy


def test_our_own_model_loading_is_not_low_vram_from_someone_else(g):
    cfg, db, clock, rig, guard = g
    rig.free_mb = 5_400  # qwen is loading: VRAM is taken but /api/ps doesn't list it yet
    with guard.scoring():
        assert not check(guard).busy
    clock.now += timedelta(seconds=10)
    assert kinds(check(guard)) == ["vram"]


def test_a_matching_process_is_busy(g):
    cfg, db, clock, rig, guard = g
    assert cfg.control.busy_process_patterns == ["gpu.bench", "RENDERJOB"]
    rig.procs.append((4242, "python.exe", r"C:\x\python.exe -m gpu.bench run --all"))
    a = check(guard)
    assert kinds(a) == ["process"] and "gpu.bench" in a.reasons[0].text
    rig.procs[-1] = (4242, "renderjob-build.exe", "")  # name match, any case
    assert kinds(check(guard)) == ["process"]
    rig.procs.pop()
    assert not check(guard).busy


def test_the_bots_own_process_never_matches(g):
    cfg, db, clock, rig, guard = g
    rig.procs.append((os.getpid(), "python.exe", "python -m pytest tests/test_RENDERJOB.py"))
    assert not check(guard).busy


def test_patterns_are_editable(g):
    cfg, db, clock, rig, guard = g
    cfg.control.busy_process_patterns = ["blender"]
    rig.procs.append((5, "Blender.exe", ""))
    assert kinds(check(guard)) == ["process"]


# --- explicit flag and lock file -------------------------------------------------------

def test_busy_flag_set_with_hours_and_cleared(g):
    cfg, db, clock, rig, guard = g
    set_flag(db, "forge benchmark", 2, clock.now)
    a = check(guard)
    assert kinds(a) == ["flag"] and "forge benchmark" in a.reasons[0].text
    clock.now += timedelta(hours=2, minutes=1)
    assert not check(guard).busy  # expired by itself
    set_flag(db, "again", None, clock.now)
    assert check(guard).busy
    assert clear_flag(db) is True and clear_flag(db) is False
    assert not check(guard).busy


def test_flag_without_hours_goes_stale_after_busy_max_hours(g, caplog):
    cfg, db, clock, rig, guard = g
    set_flag(db, "forgot to clear", None, clock.now)
    clock.now += timedelta(hours=11, minutes=59)
    assert check(guard).busy
    clock.now += timedelta(minutes=2)
    with caplog.at_level(logging.WARNING):
        assert not check(guard).busy
        check(guard)
    assert read_flag(db, clock.now, 12)[1] == "busy flag older than 12 h"
    assert sum("stale busy marker" in r.message for r in caplog.records) == 1  # logged once


def mtime_now(p, tz):
    from datetime import datetime
    return datetime.fromtimestamp(p.stat().st_mtime, tz=tz)


def test_lock_file_plain_and_json_with_a_live_owner(g):
    cfg, db, clock, rig, guard = g
    p, tz = lock_path(cfg), clock.now.tzinfo
    p.write_text("RENDERJOB benchmark\nmore text", encoding="utf-8")
    a = asyncio.run(guard.assess(now=mtime_now(p, tz)))
    assert kinds(a) == ["flag"] and "RENDERJOB benchmark" in a.reasons[0].text
    p.write_text(json.dumps({"reason": "build", "pid": 321}), encoding="utf-8")
    rig.alive = {321}
    a = asyncio.run(guard.assess(now=mtime_now(p, tz)))
    assert kinds(a) == ["flag"] and "build" in a.reasons[0].text


def test_lock_file_is_ignored_when_its_owner_is_dead_or_it_is_too_old(g):
    cfg, db, clock, rig, guard = g
    p = lock_path(cfg)
    tz = clock.now.tzinfo
    p.write_text(json.dumps({"reason": "crashed run", "pid": 999}), encoding="utf-8")
    now = mtime_now(p, tz)
    alive = lambda pid: pid in rig.alive
    assert read_lock(p, now, 12, alive) == (None, "lock gpu-busy.lock owner (PID 999) is no longer running")
    rig.alive = {999}
    assert read_lock(p, now, 12, alive) == ("crashed run", None)
    assert read_lock(p, now + timedelta(hours=12, minutes=1), 12, alive)[1] == "lock gpu-busy.lock older than 12 h"
    p.write_text("benchmark pid=555", encoding="utf-8")   # plain text naming its owner
    now = mtime_now(p, tz)
    assert read_lock(p, now, 12, alive)[0] is None and "555" in read_lock(p, now, 12, alive)[1]
    p.write_text(json.dumps({"reason": "x", "expires_at": (now - timedelta(minutes=1)).isoformat()}), encoding="utf-8")
    assert read_lock(p, now, 12, alive)[1] == "lock gpu-busy.lock expired"
    p.unlink()
    assert read_lock(p, now, 12, alive) == (None, None)


def test_a_lock_file_removed_by_its_owner_ends_the_deferral(g):
    cfg, db, clock, rig, guard = g
    p = lock_path(cfg)
    p.write_text(json.dumps({"reason": "bench", "pid": os.getpid()}), encoding="utf-8")
    rig.alive = {os.getpid()}
    clock.now = mtime_now(p, clock.now.tzinfo)
    assert kinds(check(guard)) == ["flag"]
    p.unlink()
    assert not check(guard).busy


# --- logging, per-reason time, heartbeat ----------------------------------------------

def test_a_change_is_logged_once_not_every_sample(g, caplog):
    cfg, db, clock, rig, guard = g
    rig.models = ["ministral-3:8b"]
    with caplog.at_level(logging.INFO, logger="jobbot.guard"):
        for _ in range(4):
            check(guard)
            clock.now += timedelta(seconds=30)
        rig.models = []
        check(guard)
        check(guard)
    msgs = [r.message for r in caplog.records]
    assert sum(m.startswith("GPU busy: another Ollama model") for m in msgs) == 1
    assert sum("no longer busy" in m for m in msgs) == 1


def test_busy_time_is_tracked_per_reason_and_shown_in_the_heartbeat(g):
    cfg, db, clock, rig, guard = g
    rig.models = ["ministral-3:8b"]
    check(guard)
    clock.now += timedelta(minutes=25)
    rig.procs.append((7, "python.exe", "gpu.bench"))
    check(guard)                      # a second reason starts
    clock.now += timedelta(minutes=15)
    rig.models = []
    check(guard)                      # the model reason ends
    clock.now += timedelta(minutes=5)
    rig.procs.pop()
    check(guard)
    since = clock.now - timedelta(hours=1)
    assert db.period_seconds("busy:other_model", since, clock.now) == 40 * 60
    assert db.period_seconds("busy:process", since, clock.now) == 20 * 60
    text = build_message(db, clock.now)
    assert "GPU busy (scoring waited): another model loaded 40 min · busy process 20 min" in text


def test_a_restarted_sampler_closes_periods_the_old_process_left_open(g):
    cfg, db, clock, rig, guard = g
    rig.models = ["x"]
    check(guard)
    assert db.open_period_kinds("busy:") == ["busy:other_model"]
    restarted = make_guard(cfg, db, clock, rig)

    async def go():
        task = asyncio.create_task(restarted.run())
        await asyncio.sleep(0.05)
        task.cancel()
    asyncio.run(go())
    assert db.open_period_kinds("busy:") == ["busy:other_model"]  # reopened by its first sample, only one row
    assert db.conn.execute("SELECT COUNT(*) FROM control_periods WHERE kind='busy:other_model'").fetchone()[0] == 2


# --- the bot scores as soon as the GPU is free ------------------------------------------

def test_the_idle_loop_scores_waiting_jobs_as_soon_as_the_gpu_clears(g):
    cfg, db, clock, rig, guard = g
    src, matcher, notifier = FakeSource(), FakeScorer(), RecordingNotifier()
    matcher.uses_gpu = True
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [src], notifier, matcher, clock=clock, guard=guard)
    src.jobs = [gh("1")]
    run(p)
    clock.tick()
    add_job(src)
    rig.models = ["ministral-3:8b"]
    assert run(p).held == 1

    async def go():
        async def other_project_finishes():
            await asyncio.sleep(0.15)
            rig.models = []
            await guard.check()
        task = asyncio.create_task(other_project_finishes())
        await cli._idle(p, 3.0, 0.05)  # returns after the scan interval, long after the job should be scored
        await task
    start = time.monotonic()
    asyncio.run(go())
    assert status(db, "2")["status"] == "notified" and notifier.alerts[0].score == 90
    assert time.monotonic() - start < 4


# --- models jobbot didn't load are never touched ---------------------------------------------

def test_unload_leaves_other_models_alone():
    class OtherModelOnly(FakeOllamaServer):
        def __call__(self, request):
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": "ministral-3:8b", "model": "ministral-3:8b",
                                                             "size": 6, "size_vram": 6}]})
            return super().__call__(request)

    fake = OtherModelOnly()

    async def go():
        async with client_for(fake) as c:
            return await ollama_ctl.unload(c, OWN)
    assert asyncio.run(go()) is True and fake.generate_calls == []


# --- nvidia-smi reading ----------------------------------------------------------------------

SMI = """
import sys
a = " ".join(sys.argv[1:])
if "utilization.gpu" in a:
    print("37, 2048, 9900")
elif "compute-apps" in a:
    print("11, C:\\\\Windows\\\\explorer.exe")
    print("12, C:\\\\games\\\\game.exe")
    print("13, C:\\\\games\\\\game.exe")
else:
    sys.exit(1)
"""


def test_read_gpu_parses_utilization_vram_and_process_names(tmp_path):
    script = tmp_path / "smi.py"
    script.write_text(SMI, encoding="utf-8")
    r = asyncio.run(gpu.read_gpu((sys.executable, str(script))))
    assert (r.util_pct, r.used_mb, r.free_mb) == (37, 2048, 9900) and r.procs == ["explorer.exe", "game.exe"]
    assert asyncio.run(gpu.read_gpu(("no-such-nvidia-smi",))) is None


# --- the CLI --------------------------------------------------------------------------------

def test_busy_command_sets_shows_and_clears_the_flag(tmp_path, capsys):
    cfg = make_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    assert cli.main(["--config", str(path), "busy", "--set", "forge benchmark", "--hours", "3"]) == 0
    out = capsys.readouterr().out
    assert "Busy flag set (forge benchmark)" in out and "Flag: forge benchmark" in out
    db = DB(cfg.db_path)
    assert read_flag(db, utcnow(), 12)[0] == "forge benchmark"
    db.close()
    assert cli.main(["--config", str(path), "busy", "--clear"]) == 0
    out = capsys.readouterr().out
    assert "Busy flag cleared." in out and "Flag: none" in out
    assert cli.main(["--config", str(path), "busy"]) == 0
