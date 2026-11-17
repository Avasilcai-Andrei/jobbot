from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Job:
    """One posting, as produced by any source plugin."""

    source: str                      # plugin kind, e.g. "greenhouse"
    job_id: str                      # id unique within the source
    feed: str                        # plugin instance, e.g. "greenhouse:thoughtworks"
    title: str
    company: str
    location: str
    url: str
    description: str | None = None   # plain text; None = not fetched yet
    posted_at: datetime | None = None
    remote: bool | None = None       # None = source doesn't say
    extra: dict[str, str] = field(default_factory=dict)  # source metadata worth showing the filters/LLM

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.job_id)

    @property
    def short_id(self) -> str:
        return short_id(self.source, self.job_id)


def short_id(source: str, job_id: str) -> str:
    """6 hex characters for `jobbot mark <id>`: stable, and collisions are unlikely among the alerted jobs."""
    return hashlib.sha1(f"{source}:{job_id}".encode()).hexdigest()[:6]
