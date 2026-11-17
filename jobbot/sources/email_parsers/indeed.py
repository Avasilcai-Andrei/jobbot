"""Indeed job-match emails (donotreply@match.indeed.com).

The plain-text part is a greeting, then one card (three to five lines, no links) followed by a paragraph that starts with
the "view job" link (the label is localised: "Vedeți locul de muncă: <url>"):

    <title>
    <company>
    <location>
    [salary line]

    View job: https://cts.indeed.com/v3/<long opaque token>
    Apply now: https://cts.indeed.com/v3/...

The link is a tracking redirect with no job id in it, and by design redirects are never resolved, so
`job_id` stays empty and the source falls back to a hash of company|title|location. The first link of the card is
kept as the URL: it is the only way to open the posting. No description is available.
"""
from __future__ import annotations

import re
from email.message import EmailMessage

from . import ParsedJob, parser
from ._common import body_text, clean

_CTA_RE = re.compile(r"^[^\n:]{3,60}:\s*(https://cts\.indeed\.com/\S+)\s*$")
_ANY_URL_RE = re.compile(r"https?://")


@parser("indeed")
def parse_indeed(msg: EmailMessage) -> list[ParsedJob]:
    paragraphs = [[clean(line) for line in block.splitlines() if clean(line)]
                  for block in re.split(r"\n\s*\n", body_text(msg))]
    jobs: list[ParsedJob] = []
    for card, after in zip(paragraphs, paragraphs[1:]):
        if not 3 <= len(card) <= 5 or any(_ANY_URL_RE.search(line) for line in card):
            continue
        cta = _CTA_RE.match(after[0])
        if cta is None:
            continue
        title, company, location, *extra = card   # extra: e.g. "Salariu: 110 RON - 155 RON pe ora"
        jobs.append(ParsedJob(title=title, company=company, location=location, url=cta.group(1),
                              snippet="\n".join(extra)))
    return jobs
