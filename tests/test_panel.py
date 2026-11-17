"""Control panel: what it shows (incl. why scoring is deferred) and that its keys change the control state."""
import asyncio
from datetime import timedelta

import pytest
from textual.widgets import Input, Static

from jobbot.control import Control
from jobbot.db import DB
from jobbot.panel import PanelApp, render
from jobbot.status import format_text, gather

from .conftest import make_config, make_job
from .test_pipeline import Clock


@pytest.fixture
def setup(tmp_path):
    # backend "off" and a missing nvidia-smi: the panel makes no live calls in tests
    cfg = make_config(tmp_path, llm={"backend": "off"}, control={"nvidia_smi": "no-such-nvidia-smi"})
    clock = Clock()
    db = DB(cfg.db_path)
    yield cfg, db, Control(db, cfg, clock), clock
    db.close()


def info_for(cfg, control, clock):
    return asyncio.run(gather(cfg, control, clock.now))


def add_pending(db, clock, minutes_ago, job_id="1"):
    db.insert_job(make_job(job_id), status="pending", first_seen_at=clock.now - timedelta(minutes=minutes_ago),
                  crosspost_key=f"k{job_id}")


def test_manual_deferral_is_explained_with_pending_count_and_oldest_wait(setup):
    cfg, db, control, clock = setup
    control.defer("minutes", 90)
    add_pending(db, clock, 35, "1")
    add_pending(db, clock, 5, "2")
    text = render(info_for(cfg, control, clock), clock.now)["scoring"]
    assert "manual defer" in text and "1 h 30 min left" in text
    assert "2 job(s)" in text and "waited 35 min of the 120 min hold" in text and "[unscored]" in text


def publish_busy(cfg, db, clock, rig, guard=None):
    from .gpu_fakes import make_guard
    guard = guard or make_guard(cfg, db, clock, rig)
    asyncio.run(guard.check())
    return guard


def test_gpu_busy_is_explained_with_the_reason_and_when_scoring_is_allowed(setup):
    from .gpu_fakes import Rig
    cfg, db, control, clock = setup
    control.write_bot_status(clock.now)
    add_pending(db, clock, 12)
    rig = Rig()
    rig.util = 63
    guard = publish_busy(cfg, db, clock, rig)
    info = info_for(cfg, control, clock)
    sections = render(info, clock.now)
    assert "GPU utilization was 63%" in sections["model"] and "scoring allowed from" in sections["model"]
    assert "in 10 min" in sections["model"]
    assert "auto-defer" in sections["scoring"] and "waited 12 min of the 120 min hold" in sections["scoring"]
    assert "auto-defer" in format_text(info, clock.now)  # the CLI shows it too
    rig.util = 1
    clock.tick(11)
    control.write_bot_status(clock.now)
    publish_busy(cfg, db, clock, rig, guard)
    sections = render(info_for(cfg, control, clock), clock.now)
    assert "GPU quiet for 10+ min" in sections["model"] and "auto-defer" not in sections["scoring"]


def test_gpu_status_is_hidden_when_the_bot_is_not_running(setup):
    from .gpu_fakes import Rig
    cfg, db, control, clock = setup
    publish_busy(cfg, db, clock, Rig())  # published, but the bot never wrote its beacon
    assert info_for(cfg, control, clock)["gpu"] is None


def test_paused_state_and_bot_liveness(setup):
    cfg, db, control, clock = setup
    control.pause(2)
    sections = render(info_for(cfg, control, clock), clock.now)
    assert "PAUSED" in sections["state"] and "NOT RUNNING" in sections["bot"]
    assert "manual pause" in sections["scoring"] and "bot is not running" in sections["model"]
    control.write_bot_status(clock.now)
    control.next_run_at = clock.now + timedelta(minutes=7)
    control.write_bot_status(clock.now)
    sections = render(info_for(cfg, control, clock), clock.now)
    assert "alive" in sections["bot"] and "in 7 min" in sections["bot"]
    assert "after you resume" in sections["model"]


def test_panel_reading_state_never_writes_it(setup):
    cfg, db, control, clock = setup
    control.pause(1)
    clock.tick(61 * 60)  # the pause has expired, but only the bot closes the period
    info = info_for(cfg, control, clock)
    assert info["mode"] == "running"
    assert control.read().paused is True


def test_alerts_with_marks_and_failing_feeds(setup):
    cfg, db, control, clock = setup
    for i, mins in ((1, 50), (2, 40)):
        job = make_job(str(i), title=f"Job number {i}")
        db.insert_job(job, status="notified", first_seen_at=clock.now - timedelta(minutes=mins), crosspost_key=f"k{i}")
        db.update_job(job.key, notified_at=clock.now - timedelta(minutes=mins - 1), score=80 + i)
    db.set_mark(("fake", "2"), "good", None)
    st = db.feed_state("fake:acme")
    st.consecutive_failures, st.last_error = 3, "HTTP 500"
    db.save_feed_state(st)
    sections = render(info_for(cfg, control, clock), clock.now)
    lines = sections["alerts"].splitlines()
    assert "Job number 2" in lines[0] and "good" in lines[0] and "Job number 1" in lines[1]
    assert make_job("2").short_id in lines[0]
    assert "fake:acme (3×)" in sections["last"]


def test_keys_change_the_control_state_and_footer_lists_them(setup):
    cfg, db, control, clock = setup
    unloads = []

    async def fake_unload(_cfg):
        unloads.append(1)
        return 0

    async def go():
        app = PanelApp(cfg, db=DB(cfg.db_path), unload=fake_unload, clock=clock)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "RUNNING" in str(app.query_one("#state", Static).render())
            await pilot.press("d")
            assert control.read().defer_kind == "next_scan"
            await pilot.press("r")
            assert control.read().mode == "running"
            await pilot.press("p")
            assert control.read().paused and unloads == [1]
            await pilot.pause()
            assert "PAUSED" in str(app.query_one("#state", Static).render())
            await pilot.press("p")
            assert not control.read().paused
            await pilot.press("m")
            await pilot.pause()
            app.screen.query_one(Input).value = "30"
            await pilot.press("enter")
            await pilot.pause()
            s = control.read()
            assert s.defer_kind == "minutes" and s.defer_until == clock.now + timedelta(minutes=30)
            await pilot.press("u")
            assert unloads == [1, 1]
            bindings = {b.key: b.description for b in app.BINDINGS}
            assert bindings["p"] == "Pause/Resume" and bindings["q"] == "Quit panel"
    asyncio.run(go())
