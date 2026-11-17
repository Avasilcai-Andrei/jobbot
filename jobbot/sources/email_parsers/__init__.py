"""Per-sender parsers for job-alert emails. One parser turns one email into a list of ParsedJob.

Parsers are pure functions of the already-decoded message (no network, no page fetches), so they are tested
against .eml fixtures in tests/fixtures/email/.
"""
from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from typing import Callable

# From-domain -> parser name. A domain also matches its subdomains (mail.linkedin.com -> linkedin.com).
# Overridable per mailbox with `senders:` in config.yaml.
DEFAULT_SENDERS: dict[str, str] = {
    "linkedin.com": "linkedin",
    "ejobs.ro": "ejobs",
    "bestjobs.eu": "bestjobs",
    "hipo.ro": "hipo",
    "indeed.com": "indeed",
    "glassdoor.com": "glassdoor",
    "jobtogether.com": "jobtogether",
}


@dataclass
class ParsedJob:
    title: str
    company: str
    location: str
    url: str                    # tracking parameters already stripped
    job_id: str | None = None   # the platform's own id when the URL has one; otherwise a hash is used
    snippet: str = ""           # any description text the email carries


Parser = Callable[[EmailMessage], list[ParsedJob]]
PARSERS: dict[str, Parser] = {}


def parser(name: str) -> Callable[[Parser], Parser]:
    def deco(fn: Parser) -> Parser:
        PARSERS[name] = fn
        return fn
    return deco


def match_sender(domain: str, senders: dict[str, str]) -> tuple[str, str] | None:
    """(configured domain, parser name) for a From domain, or None for a sender that is not in the table.
    The parser may not be written yet (name not in PARSERS): the caller must then leave the mail unrecorded."""
    domain = domain.lower().strip(".")
    for known, name in senders.items():
        if domain == known or domain.endswith("." + known):
            return known, name
    return None


# Importing the parser modules registers them in PARSERS (they import `parser` and `ParsedJob` from here).
from . import indeed, linkedin  # noqa: E402,F401
