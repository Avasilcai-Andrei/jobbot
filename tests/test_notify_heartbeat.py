import asyncio
import smtplib
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from jobbot import heartbeat
from jobbot.app import channels, make_notifier
from jobbot.cli import EXIT_CONFIG, acquire_lock, main
from jobbot.config import Secrets
from jobbot.db import DB
from jobbot.notify import Alert, ConsoleNotifier, EmailNotifier, NotifierChain, NotifyError, email_for_alert

from .conftest import make_config, make_job, mock_client

NOW = datetime(2026, 10, 30, 7, 30, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def alert(**kw):
    defaults = dict(job=make_job(), score=82, reason="Fits a junior data role.", missing=["Airflow"],
                    location_rule="Cluj-Napoca", low_priority=False, first_seen_at=NOW)
    return Alert(**{**defaults, **kw})


class FakeSMTP:
    """Stands in for smtplib.SMTP_SSL; records what would have been sent."""
    sent: list = []
    fail_login = False

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        if FakeSMTP.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
        self.user = user

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


@pytest.fixture(autouse=True)
def _reset_smtp():
    FakeSMTP.sent, FakeSMTP.fail_login = [], False


def email(**kw):
    return EmailNotifier("me@gmail.com", "abcdabcdabcdabcd", "me@gmail.com", smtp_factory=FakeSMTP, **kw)


# --- email ------------------------------------------------------------------------

def test_email_alert_content():
    subject, body = email_for_alert(alert())
    assert subject == "[jobbot] 82/100: Junior Data Engineer @ Acme"
    assert "Location: Cluj-Napoca, Romania (Cluj-Napoca)" in body and "  - Airflow" in body
    sid = make_job().short_id
    assert body.rstrip().endswith(f"-- jobbot (fake:acme, id {sid})") and "https://example.com/jobs/1" in body
    assert f"python -m jobbot mark {sid} good|bad" in body
    assert email_for_alert(alert(score=None, unscored="backend off"))[0].startswith("[jobbot] [unscored] unscored:")


def test_email_send_and_errors_never_show_the_password():
    run(email().send_alert(alert()))
    msg = FakeSMTP.sent[0]
    assert msg["To"] == "me@gmail.com" and msg["Subject"].startswith("[jobbot] 82/100")
    run(email().send_text("💓 jobbot daily check\nline 2"))
    assert FakeSMTP.sent[1]["Subject"] == "[jobbot] 💓 jobbot daily check"
    FakeSMTP.fail_login = True
    with pytest.raises(NotifyError) as e:
        run(email().send_text("x"))
    assert "code 535" in str(e.value) and "abcd" not in str(e.value)


# --- chain ------------------------------------------------------------------------

class Recorder:
    def __init__(self, fail=False):
        self.fail, self.got = fail, []

    async def send_alert(self, a):
        if self.fail:
            raise NotifyError("down")
        self.got.append(a)

    async def send_text(self, t):
        if self.fail:
            raise NotifyError("down")
        self.got.append(t)


def test_fallback_uses_the_next_channel_only_when_needed():
    d, e = Recorder(), Recorder()
    run(NotifierChain([("Discord", d), ("Email", e)]).send_text("hi"))
    assert d.got == ["hi"] and e.got == []
    d.fail = True
    run(NotifierChain([("Discord", d), ("Email", e)]).send_alert(alert()))
    assert len(e.got) == 1
    e.fail = True
    with pytest.raises(NotifyError, match="Discord: down; Email: down"):
        run(NotifierChain([("Discord", d), ("Email", e)]).send_text("hi"))


def test_both_mode_sends_everywhere_and_succeeds_if_one_works():
    d, e = Recorder(), Recorder()
    run(NotifierChain([("Discord", d), ("Email", e)], mode="both").send_text("hi"))
    assert d.got == e.got == ["hi"]
    d.fail = True
    run(NotifierChain([("Discord", d), ("Email", e)], mode="both").send_text("again"))  # no exception
    assert e.got == ["hi", "again"]


def test_channels_from_config(tmp_path):
    cfg = make_config(tmp_path)
    http = mock_client(lambda r: httpx.Response(204))
    assert channels(cfg, http) == [] and isinstance(make_notifier(cfg, http, console=False), ConsoleNotifier)
    cfg.secrets = Secrets(discord_webhook_url="https://discord.com/api/webhooks/1/x",
                          smtp_user="me@gmail.com", smtp_app_password="pw", alert_email_to="me@gmail.com")
    assert [n for n, _ in channels(cfg, http)] == ["Discord", "Email"]
    chain = make_notifier(cfg, http, console=False)
    assert isinstance(chain, NotifierChain) and chain.mode == "fallback"
    cfg.notify.email.mode = "both"
    assert make_notifier(cfg, http, console=False).mode == "both"
    cfg.notify.email.mode = "off"
    assert [n for n, _ in channels(cfg, http)] == ["Discord"]


def test_app_password_spaces_are_removed(tmp_path, monkeypatch):
    from jobbot.config import load_config
    (tmp_path / "config.yaml").write_text("profile: {roles: []}\nlocations: {rules: []}\n", encoding="utf-8")
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "qrst uvwx yzab cdef")
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    s = load_config(tmp_path / "config.yaml").secrets
    assert s.smtp_app_password == "qrstuvwxyzabcdef" and s.alert_email_to == "me@gmail.com"


# --- heartbeat ----------------------------------------------------------------------

def test_heartbeat_due():
    local = datetime(2026, 10, 30, 9, 5)
    assert heartbeat.due(local, "09:00", None)
    assert not heartbeat.due(local, "09:00", "2026-10-30")          # already sent today
    assert heartbeat.due(local, "09:00", "2026-10-29")
    assert not heartbeat.due(datetime(2026, 10, 30, 8, 59), "09:00", None)
    assert heartbeat.due(datetime(2026, 10, 30, 23, 0), "09:00", "2026-10-29")  # PC was off at 9: send late


def test_heartbeat_message(tmp_path):
    db = DB(":memory:")
    for i in range(3):
        start = NOW - timedelta(minutes=10 * (3 - i))
        db.record_run(start, start + timedelta(seconds=30),
                      {"fetched": 1300 + i, "new": 2, "passed_rules": 1, "scored": 1, "notified": i % 2,
                       "unscored": 0, "rescored": 0})
    db.record_run(NOW - timedelta(days=2), NOW - timedelta(days=2), {"fetched": 99})  # older than 24 h
    db.record_run(NOW - timedelta(minutes=5), NOW - timedelta(minutes=5), {"crashed": 1})
    st = db.feed_state("workday:ing/ICSGBLCOR")
    st.consecutive_failures, st.last_error = 2, "HTTP 503 from https://ing.wd3.myworkdayjobs.com/..."
    db.save_feed_state(st)
    text = heartbeat.build_message(db, NOW)
    assert text.splitlines()[0] == "💓 jobbot daily check (last 24 h): 4 runs, last 5 min ago, 1 crashed."
    assert "Postings fetched: 1,301 per run, 6 new." in text
    assert "Passed rules: 3 · scored: 3 · alerted: 1." in text
    assert "Feeds failing: workday:ing/ICSGBLCOR (2× HTTP 503" in text
    assert "⚠️ No runs" in heartbeat.build_message(DB(":memory:"), NOW)


def test_heartbeat_sent_once_a_day_and_retried_after_failure(tmp_path):
    cfg = make_config(tmp_path)
    db = DB(":memory:")
    rec = Recorder(fail=True)
    morning = datetime(2026, 10, 30, 9, 1)
    assert not run(heartbeat.maybe_send(cfg, db, rec, NOW, morning))     # send failed: not marked as sent
    rec.fail = False
    assert run(heartbeat.maybe_send(cfg, db, rec, NOW, morning))
    assert not run(heartbeat.maybe_send(cfg, db, rec, NOW, morning.replace(hour=15)))   # same day
    assert run(heartbeat.maybe_send(cfg, db, rec, NOW, morning + timedelta(days=1)))
    assert len(rec.got) == 2 and db.get_meta(heartbeat.META_KEY) == "2026-10-31"
    cfg.notify.heartbeat.enabled = False
    assert not run(heartbeat.maybe_send(cfg, db, rec, NOW, morning + timedelta(days=2)))


def test_heartbeat_time_is_validated(tmp_path):
    with pytest.raises(ValueError):
        make_config(tmp_path, notify={"heartbeat": {"time": "9am"}})


# --- run command --------------------------------------------------------------------

def test_run_refuses_without_a_notification_channel(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("profile: {roles: []}\nlocations: {rules: []}\n", encoding="utf-8")
    for k in ("DISCORD_WEBHOOK_URL", "SMTP_USER", "SMTP_APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert main(["--config", str(tmp_path / "config.yaml"), "run"]) == EXIT_CONFIG


def test_single_instance_lock(tmp_path):
    first = acquire_lock(tmp_path / "jobbot.lock")
    assert first is not None
    assert acquire_lock(tmp_path / "jobbot.lock") is None
    first.close()
    second = acquire_lock(tmp_path / "jobbot.lock")
    assert second is not None
    second.close()


def test_unedited_env_example_values_are_ignored(tmp_path, monkeypatch):
    from jobbot.config import load_config
    (tmp_path / "config.yaml").write_text("profile: {roles: []}\nlocations: {rules: []}\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/<id>/<token>")
    monkeypatch.setenv("SMTP_USER", "you@gmail.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "abcd efgh ijkl mnop")
    s = load_config(tmp_path / "config.yaml").secrets
    assert s.discord_webhook_url is None and s.smtp_user is None and s.smtp_app_password is None


# --- mark / report --marks -----------------------------------------------------------

def test_mark_alerts_and_list_them(tmp_path, capsys):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("profile: {roles: []}\nlocations: {rules: []}\n", encoding="utf-8")
    db = DB(tmp_path / "data" / "jobbot.sqlite3")
    jobs = [make_job("1", title="Junior Data Engineer"), make_job("2", title="Python Developer"),
            make_job("3", title="Data Analyst")]
    for i, job in enumerate(jobs):
        db.insert_job(job, status="new", first_seen_at=NOW, crosspost_key=f"k{i}")
    db.update_job(jobs[0].key, status="notified", score=82, notified_at=NOW)
    db.update_job(jobs[1].key, status="notified", notified_at=NOW + timedelta(hours=1))  # unscored alert
    db.close()  # jobs[2] was never alerted
    cli = lambda *a: main(["--config", str(cfg_path), *a])  # noqa: E731

    assert cli("mark", jobs[0].short_id.upper(), "good") == 0
    assert cli("mark", f"#{jobs[1].short_id}", "bad", "embedded", "C++,", "not", "my", "field") == 0
    assert cli("mark", jobs[1].short_id, "good") == 0  # changing your mind overwrites
    assert cli("mark", jobs[1].short_id, "bad", "embedded C++") == 0
    out = capsys.readouterr().out
    assert f"Marked {jobs[0].short_id} good: Junior Data Engineer @ Acme (score 82" in out
    assert cli("mark", jobs[2].short_id, "good") == 1  # not an alert
    assert cli("mark", "zzzzzz", "bad") == 1
    assert "report --marks" in capsys.readouterr().out

    assert cli("report", "--marks") == 0
    out = capsys.readouterr().out
    assert "Alerts: 2; marked 2 (1 good, 1 bad), not marked yet 0." in out
    line = next(l for l in out.splitlines() if l.startswith(jobs[1].short_id))
    assert "bad" in line and "Python Developer @ Acme" in line and line.endswith("embedded C++") and " - " in line
    assert jobs[2].short_id not in out


def test_marks_report_lists_unmarked_alerts(tmp_path):
    from jobbot.report import build_marks_report

    db = DB(":memory:")
    assert build_marks_report(db) == "No alerts sent yet."
    job = make_job("9", title="ML Intern")
    db.insert_job(job, status="new", first_seen_at=NOW, crosspost_key="k")
    db.update_job(job.key, status="notified", score=70, notified_at=NOW)
    out = build_marks_report(db)
    assert "not marked yet 1" in out and "Not marked yet" in out
    assert any(l.startswith(job.short_id) and "ML Intern @ Acme" in l for l in out.splitlines())
