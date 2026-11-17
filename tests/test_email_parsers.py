"""Email parsers: LinkedIn and Indeed, against redacted .eml fixtures. No network."""
from __future__ import annotations

import email
import email.policy
import os
from email.message import EmailMessage
from pathlib import Path

import pytest

from jobbot.sources.email_parsers import PARSERS

FIXTURES = Path(__file__).parent / "fixtures" / "email"


def load(name: str) -> EmailMessage:
    with (FIXTURES / name).open("rb") as f:
        return email.message_from_binary_file(f, policy=email.policy.default)  # type: ignore[return-value]


def plain(text: str) -> EmailMessage:
    m = EmailMessage()
    m["From"] = "x@example.com"
    m.set_content(text)
    return m


def test_fixtures_are_redacted():
    for name in ("linkedin.eml", "indeed.eml"):
        raw = (FIXTURES / name).read_text(encoding="utf-8", errors="replace").lower()
        # JOBBOT_PRIVATE_TERMS (comma-separated, e.g. your name and address) adds personal strings to check locally.
        private = [t.strip().lower() for t in os.environ.get("JOBBOT_PRIVATE_TERMS", "").split(",") if t.strip()]
        for secret in ("@gmail.com", "otptoken", "trackingid", "delivered-to", *private):
            assert secret not in raw, f"{name} still contains {secret}"


def test_linkedin_digest():
    jobs = PARSERS["linkedin"](load("linkedin.eml"))
    assert [j.job_id for j in jobs] == ["4377951132", "4348957636", "4417944629", "4438343062",
                                        "4439530833", "4439305158"]
    first, second = jobs[0], jobs[1]
    assert (first.title, first.company, first.location) == ("Software Engineer (Java) - Core Platform",
                                                            "Revolut", "Romania")
    # a card with a badge line, and the digest header line, must not leak into the fields
    assert (second.title, second.company, second.location) == ("Embedded Software Apps Engineering Intern",
                                                               "Analog Devices", "Cluj-Napoca")
    assert jobs[4].location == "Cluj-Napoca Metropolitan Area"   # a card with no badge line
    assert all(j.url == f"https://www.linkedin.com/jobs/view/{j.job_id}/" for j in jobs)
    assert all("?" not in j.url and "trackingId" not in j.url for j in jobs)


def test_linkedin_unknown_template_gives_nothing():
    assert PARSERS["linkedin"](plain("Hi! Someone viewed your profile.\n\nSee who: https://www.linkedin.com/x")) == []
    # a link with no card above it is not a job
    assert PARSERS["linkedin"](plain("Your job alert for x in y\nView job: https://www.linkedin.com/comm/jobs/view/1/")) == []


def test_linkedin_repeated_card_counts_once():
    card = "Data Engineer\nAcme\nCluj-Napoca\nView job: https://www.linkedin.com/comm/jobs/view/42/?trk=a\n"
    jobs = PARSERS["linkedin"](plain(card + "-" * 20 + "\n" + card))
    assert [j.job_id for j in jobs] == ["42"]


def test_indeed_single_job():
    jobs = PARSERS["indeed"](load("indeed.eml"))
    assert len(jobs) == 1
    j = jobs[0]
    assert (j.title, j.company, j.location) == ("Internship Data Engineer - Sportsbet, Hybrid",
                                                "Flutter Entertainment", "Cluj-Napoca")
    assert j.url.startswith("https://cts.indeed.com/v3/")
    assert j.job_id is None   # the link is an opaque redirect; the source hashes company|title|location


def test_indeed_ignores_footer_and_greeting_blocks():
    text = ("Hello,\n\nGreat match.\n\nData Engineer\nAcme\nCluj-Napoca\n\n"
            "View job: https://cts.indeed.com/v3/AAA\nApply now: https://cts.indeed.com/v3/BBB\n\n"
            "Keep your profile current\nName\nEdit profile: https://cts.indeed.com/v3/CCC\n\n"
            "Indeed\nHelp: https://cts.indeed.com/v3/DDD\n")
    jobs = PARSERS["indeed"](plain(text))
    assert [(j.title, j.url) for j in jobs] == [("Data Engineer", "https://cts.indeed.com/v3/AAA")]


def test_indeed_unknown_template_gives_nothing():
    assert PARSERS["indeed"](plain("Your account was updated.\n\nSee: https://cts.indeed.com/v3/A")) == []


def test_indeed_salary_line_goes_to_snippet():
    text = ("Hello,\n\nGreat match.\n\nAI Tool R&D\nAbroadWorks Inc.\nRomania\nSalary: 110 RON - 155 RON per hour\n\n"
            "View job: https://cts.indeed.com/v3/AAA\n")
    [j] = PARSERS["indeed"](plain(text))
    assert (j.title, j.company, j.location) == ("AI Tool R&D", "AbroadWorks Inc.", "Romania")
    assert j.snippet == "Salary: 110 RON - 155 RON per hour"


def test_linkedin_two_header_lines_and_badges():
    text = ("Your job alert for Generative AI Engineer in Cluj-Napoca\n3 new jobs match your preferences.\n\n"
            "AI Solution Expert\nNTT DATA\nCluj-Napoca\n\nThis company is actively hiring\nApply with resume & profile\n"
            "View job: https://www.linkedin.com/comm/jobs/view/7/?trk=x\n\n" + "-" * 30 + "\n\n"
            "Python Engineer\nCiklum\nRomania\nView job: https://www.linkedin.com/comm/jobs/view/8/?trk=x\n")
    jobs = PARSERS["linkedin"](plain(text))
    assert [(j.title, j.company, j.location) for j in jobs] == [
        ("AI Solution Expert", "NTT DATA", "Cluj-Napoca"), ("Python Engineer", "Ciklum", "Romania")]
