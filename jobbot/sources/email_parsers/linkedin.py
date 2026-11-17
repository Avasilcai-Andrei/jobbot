"""LinkedIn job-alert digests (jobalerts-noreply@linkedin.com).

The plain-text part is a list of cards separated by a line of dashes. Each card is

    <title>
    <company>
    <location>
    [badge lines: "This company is actively hiring", "101 school alumni", ...]
    View job: https://www.linkedin.com/comm/jobs/view/<id>/?<tracking>

so the card is read backwards from its "View job" line: the first three non-empty lines are title, company and
location. The tracking query is dropped and the URL is rewritten to the canonical /jobs/view/<id>/. The page is
never fetched: the email carries no description.
"""
from __future__ import annotations

import re
from email.message import EmailMessage

from . import ParsedJob, parser
from ._common import body_text, clean

_VIEW_RE = re.compile(r"^[^\n:]{3,40}:\s*(https://(?:www\.|ro\.)?linkedin\.com/(?:comm/)?jobs/view/(\d+)\S*)\s*$")
# "Your job alert for <search> in <place>" and "3 new jobs match your preferences." (some digests have both).
_HEADER_RE = re.compile(r"^(your job alert for\b|\d+\+? new jobs?\b)", re.I)
_SEPARATOR_RE = re.compile(r"^-{10,}\s*$")


@parser("linkedin")
def parse_linkedin(msg: EmailMessage) -> list[ParsedJob]:
    jobs: list[ParsedJob] = []
    seen: set[str] = set()
    card: list[str] = []
    for raw in body_text(msg).splitlines():
        if _SEPARATOR_RE.match(raw):
            card = []
            continue
        m = _VIEW_RE.match(raw.strip())
        if m is None:
            line = clean(raw)
            if line:
                card.append(line)
            continue
        job_id = m.group(2)
        # The first card is preceded by one or two header lines. A card needs a title,
        # a company and a location; anything shorter is a template we don't understand and is skipped.
        while card and _is_header(card[0]):
            card = card[1:]
        if len(card) >= 3 and job_id not in seen:
            seen.add(job_id)
            title, company, location = card[:3]
            jobs.append(ParsedJob(title=title, company=company, location=location,
                                  url=f"https://www.linkedin.com/jobs/view/{job_id}/", job_id=job_id))
        card = []
    return jobs


def _is_header(line: str) -> bool:
    """The digest's opening line, which is not part of any card."""
    return bool(_HEADER_RE.match(line))
