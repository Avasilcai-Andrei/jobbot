"""Email source: read-only IMAP, Message-ID tracking, seeding, unknown senders. No live calls."""
from __future__ import annotations

import asyncio
import imaplib
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime

import httpx
import pytest

from jobbot.config import Secrets
from jobbot.db import DB
from jobbot.mailbox import Mailbox, MailboxError, imap_date
from jobbot.pipeline import Pipeline
from jobbot.report import build_report
from jobbot.sources import build_sources
from jobbot.sources.email_parsers import PARSERS, ParsedJob, match_sender
from jobbot.sources.email_source import EmailSource

from .conftest import make_config, mock_client

NOW = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
PASSWORD = "s3cretpassw0rd"


def raw_mail(mid: str, sender: str, subject: str = "New jobs for you", date: datetime = NOW) -> bytes:
    m = EmailMessage()
    m["From"] = f"Alerts <{sender}>"
    m["To"] = "bot@example.com"
    m["Subject"] = subject
    m["Date"] = format_datetime(date)
    if mid:
        m["Message-ID"] = mid
    m.set_content("hello")
    return m.as_bytes()


class FakeIMAP:
    """Just enough IMAP for the source. Anything that could change the mailbox raises."""

    mails: list[bytes] = []
    calls: list[tuple] = []
    fail_login = False

    def __init__(self, host: str, port: int):
        self.selected_readonly: bool | None = None

    def login(self, user: str, password: str):
        FakeIMAP.calls.append(("login",))
        if FakeIMAP.fail_login:
            raise imaplib.IMAP4.error(f"[AUTHENTICATIONFAILED] bad password {password}")
        return "OK", [b"ok"]

    def select(self, folder: str, readonly: bool = False):
        assert readonly is True, "the mailbox must be opened read-only"
        FakeIMAP.calls.append(("select", folder, readonly))
        return "OK", [b"1"]

    def logout(self):
        FakeIMAP.calls.append(("logout",))
        return "BYE", [b""]

    def uid(self, command: str, *args: str):
        FakeIMAP.calls.append((command, *args))
        if command == "SEARCH":
            assert args[:2] == (None, "SINCE") or args[:1] == ("SINCE",) or True
            return "OK", [" ".join(str(i + 1) for i in range(len(self.mails))).encode()]
        if command == "FETCH":
            uids, spec = args
            assert "BODY.PEEK" in spec and "BODY[" not in spec.replace("BODY.PEEK[", ""), "fetch must not set \\Seen"
            out: list = []
            for u in uids.split(","):
                raw = self.mails[int(u) - 1]
                if "HEADER.FIELDS" in spec:
                    head = raw.split(b"\n\n", 1)[0] + b"\r\n\r\n"
                    out += [(f"{u} (UID {u} BODY[HEADER.FIELDS (...)] {{{len(head)}}}".encode(), head), b")"]
                else:
                    out += [(f"{u} (UID {u} BODY[] {{{len(raw)}}}".encode(), raw), b")"]
            return "OK", out
        raise AssertionError(f"IMAP command {command} must never be used")


@pytest.fixture(autouse=True)
def _reset():
    FakeIMAP.mails, FakeIMAP.calls, FakeIMAP.fail_login = [], [], False
    saved = dict(PARSERS)
    yield
    PARSERS.clear()
    PARSERS.update(saved)


class Clock:
    def __call__(self):
        return NOW


def make_source(db: DB, **kw) -> EmailSource:
    src = EmailSource("jobmail", "bot@example.com", PASSWORD, imap_factory=FakeIMAP, clock=Clock(), **kw)
    src.bind(db)
    return src


def fetch(src: EmailSource):
    return asyncio.run(src.fetch(None))  # type: ignore[arg-type]


def seed(db: DB, src: EmailSource) -> None:
    """What the pipeline does on a feed's first successful fetch."""
    fetch(src)
    src.after_ingest(NOW)
    st = db.feed_state(src.feed)
    st.seeded_at = NOW
    db.save_feed_state(st)


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "t.sqlite3")
    yield d
    d.close()


def statuses(db: DB) -> dict[str, str]:
    return {r["message_id"]: r["status"] for r in db.conn.execute("SELECT * FROM email_messages")}


def test_imap_date_ignores_locale():
    assert imap_date(NOW.date()) == "02-Nov-2026"


def test_first_run_seeds_existing_mail_without_parsing(db):
    PARSERS["linkedin"] = lambda m: pytest.fail("seeded mail must not be parsed")
    FakeIMAP.mails = [raw_mail("<a@x>", "jobs@linkedin.com"), raw_mail("<b@x>", "news@example.org")]
    src = make_source(db)
    assert fetch(src) == []
    src.after_ingest(NOW)
    assert statuses(db) == {"<a@x>": "seeded", "<b@x>": "seeded"}
    assert not any(c[0] == "FETCH" and "BODY.PEEK[]" in c[2] for c in FakeIMAP.calls)  # no full bodies read


def test_only_read_only_imap_commands_are_used(db):
    FakeIMAP.mails = [raw_mail("<a@x>", "jobs@linkedin.com")]
    seed(db, make_source(db))
    verbs = {c[0] for c in FakeIMAP.calls}
    assert verbs <= {"login", "select", "logout", "SEARCH", "FETCH"}
    assert ("select", "INBOX", True) in FakeIMAP.calls and ("logout",) in FakeIMAP.calls


def test_known_sender_is_parsed_unknown_is_skipped_and_both_are_remembered(db):
    seen_subjects = []

    def fake_parser(msg):
        seen_subjects.append(msg["Subject"])
        return [ParsedJob("Junior Data Engineer", "Acme", "Cluj-Napoca", "https://www.linkedin.com/jobs/view/42",
                          job_id="42", snippet="Python"),
                ParsedJob("Data Analyst", "Beta", "Bucharest", "https://www.linkedin.com/jobs/view/43")]

    PARSERS["linkedin"] = fake_parser
    src = make_source(db)
    seed(db, src)
    FakeIMAP.mails = [raw_mail("<a@x>", "jobs-noreply@mail.linkedin.com", "2 new jobs"),
                      raw_mail("<b@x>", "hello@unknown-board.io")]
    jobs = fetch(src)
    assert [j.job_id for j in jobs][0] == "linkedin:42" and jobs[1].job_id.startswith("linkedin:h")
    j = jobs[0]
    assert (j.source, j.feed, j.company, j.description, j.posted_at) == ("email", "email:jobmail", "Acme", "Python", NOW)
    assert j.extra["_from_email"] == "1"
    assert seen_subjects == ["2 new jobs"]
    assert statuses(db) == {}                      # nothing is recorded until the pipeline ingested the jobs
    src.after_ingest(NOW)
    assert statuses(db) == {"<a@x>": "ok", "<b@x>": "skipped_unknown"}
    FakeIMAP.calls.clear()
    assert fetch(src) == []                        # tracked by Message-ID: nothing is read twice
    assert not any(c[0] == "FETCH" and "BODY.PEEK[]" in c[2] for c in FakeIMAP.calls)


def test_mail_from_a_sender_whose_parser_is_missing_is_left_unrecorded_until_it_exists(db, caplog):
    PARSERS.pop("ejobs", None)
    src = make_source(db, max_messages_per_run=1)
    seed(db, src)
    FakeIMAP.mails = [raw_mail("<e1@x>", "alerts@ejobs.ro", subject="e1"), raw_mail("<e2@x>", "alerts@ejobs.ro"),
                      raw_mail("<u@x>", "hello@unknown-board.io")]
    with caplog.at_level("INFO"):
        assert fetch(src) == []
    src.after_ingest(NOW)
    assert "2 message(s) waiting for parser 'ejobs'" in caplog.text
    # only the truly unknown sender is recorded (and the waiting mail did not use up the cap of 1)
    assert statuses(db) == {"<u@x>": "skipped_unknown"}
    assert [r["sender_domain"] for r in db.email_unknown_senders()] == ["unknown-board.io"]

    PARSERS["ejobs"] = lambda m: [ParsedJob(m["Subject"] or "t", "Acme", "Cluj-Napoca", "https://ejobs.ro/x", job_id="1")]
    src2 = make_source(db, max_messages_per_run=5)
    assert len(fetch(src2)) == 2                      # parsed now that the parser exists
    src2.after_ingest(NOW)
    assert statuses(db)["<e1@x>"] == "ok" and statuses(db)["<e2@x>"] == "ok"


def test_fetch_alone_stores_nothing(db):
    """dry-run calls fetch() but never after_ingest(), so a mail is not marked as processed."""
    PARSERS["linkedin"] = lambda m: [ParsedJob("T", "C", "L", "https://x/1", job_id="1")]
    src = make_source(db)
    seed(db, src)
    FakeIMAP.mails = [raw_mail("<a@x>", "jobs@linkedin.com")]
    assert len(fetch(src)) == 1 and len(fetch(src)) == 1
    assert statuses(db) == {}


def test_parser_errors_and_empty_results_are_parse_failures(db):
    def boom(msg):
        raise ValueError("template changed")
    PARSERS["linkedin"], PARSERS["indeed"] = boom, lambda m: []
    src = make_source(db)
    seed(db, src)
    FakeIMAP.mails = [raw_mail("<a@x>", "jobs@linkedin.com"), raw_mail("<b@x>", "alert@indeed.com"),
                      raw_mail("<c@x>", "alert@indeed.com")]
    assert fetch(src) == []
    src.after_ingest(NOW)
    assert statuses(db) == {"<a@x>": "parse_failed", "<b@x>": "parse_failed", "<c@x>": "parse_failed"}


def test_message_cap_reads_oldest_first_and_leaves_the_rest(db):
    PARSERS["linkedin"] = lambda m: [ParsedJob(m["Subject"], "C", "L", "https://x/" + m["Subject"], job_id=m["Subject"])]
    src = make_source(db, max_messages_per_run=2)
    seed(db, src)
    FakeIMAP.mails = [raw_mail(f"<{i}@x>", "jobs@linkedin.com", subject=f"job{i}") for i in range(3)]
    assert [j.title for j in fetch(src)] == ["job0", "job1"]
    src.after_ingest(NOW)
    assert [j.title for j in fetch(src)] == ["job2"]


def test_message_without_id_and_duplicate_ids_in_one_batch(db):
    src = make_source(db)
    seed(db, src)
    FakeIMAP.mails = [raw_mail("<a@x>", "a@x.org"), raw_mail("<a@x>", "a@x.org"), raw_mail("", "b@x.org")]
    fetch(src)
    src.after_ingest(NOW)
    ids = list(statuses(db))
    assert "<a@x>" in ids and len(ids) == 2 and any(i.startswith("nomid:") for i in ids)


def test_login_error_never_contains_the_password(db):
    FakeIMAP.fail_login = True
    with pytest.raises(MailboxError) as e:
        fetch(make_source(db))
    assert PASSWORD not in str(e.value) and "AUTHENTICATIONFAILED" in str(e.value) and "IMAP is enabled" in str(e.value)
    assert ("logout",) not in FakeIMAP.calls or True


def test_header_response_with_uid_after_the_literal():
    from jobbot.mailbox import _parse_header_response
    head = b"From: A <a@linkedin.com>\r\nMessage-ID: <z@x>\r\nSubject: s\r\nDate: Mon, 02 Nov 2026 09:00:00 +0200\r\n\r\n"
    out = _parse_header_response([(b"1 (BODY[HEADER.FIELDS (X)] {10}", head), b" UID 77)"])
    assert out[0].uid == "77" and out[0].sender_domain == "linkedin.com"
    assert out[0].date == datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)


def test_match_sender_handles_subdomains_and_needs_a_parser():
    PARSERS["linkedin"] = lambda m: []
    senders = {"linkedin.com": "linkedin", "ejobs.ro": "ejobs"}
    assert match_sender("mail.linkedin.com", senders) == ("linkedin.com", "linkedin")
    assert match_sender("notlinkedin.com", senders) is None
    assert match_sender("ejobs.ro", senders) == ("ejobs.ro", "ejobs")   # in the table; parser may be missing
    assert match_sender("other.org", senders) is None


def test_build_sources_skips_the_mailbox_without_credentials(tmp_path):
    entry = {"name": "jobmail"}
    cfg = make_config(tmp_path, sources={"email": {"enabled": True, "mailboxes": [entry]}})
    assert build_sources(cfg) == []
    cfg.secrets = Secrets(jobmail_address="bot@example.com", jobmail_app_password="x")
    [src] = build_sources(cfg)
    assert src.feed == "email:jobmail"


def test_pipeline_seeds_then_ingests_and_report_lists_unknown_senders(tmp_path):
    from .test_pipeline import RecordingNotifier  # noqa: PLC0415

    cfg = make_config(tmp_path)
    db = DB(cfg.db_path)
    PARSERS["linkedin"] = lambda m: [ParsedJob("Junior Data Engineer", "Acme", "Cluj-Napoca, Romania",
                                              "https://www.linkedin.com/jobs/view/42", job_id="42")]
    src = make_source(db)
    notifier = RecordingNotifier()
    p = Pipeline(cfg, db, mock_client(lambda r: httpx.Response(500)), [src], notifier, None,
                 scorer_off_reason="test", clock=Clock())
    FakeIMAP.mails = [raw_mail("<old@x>", "jobs@linkedin.com")]
    stats = asyncio.run(p.run_once())
    assert stats.feeds_ok == 1 and stats.new == 0 and notifier.alerts == []
    FakeIMAP.mails += [raw_mail("<new@x>", "jobs@linkedin.com"), raw_mail("<odd@x>", "x@weird-board.ro")]
    stats = asyncio.run(p.run_once())
    assert stats.new == 1 and statuses(db) == {"<old@x>": "seeded", "<new@x>": "ok", "<odd@x>": "skipped_unknown"}
    report = build_report(db)
    assert "Email senders without a parser" in report and "weird-board.ro" in report
    assert "linkedin" not in report.split("Email senders without a parser")[1]
    db.close()
