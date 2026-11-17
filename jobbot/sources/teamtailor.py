"""Teamtailor career-site RSS feed: https://support.teamtailor.com/en/articles/11171756-rss-feed-how-to-guide

GET https://{site}/jobs.rss (site = {company}.teamtailor.com or the company's own career domain). Teamtailor says
the feed data is public. It carries the full description, so no detail requests are needed. 100 jobs per page by
default; `offset` and `per_page` page through larger boards.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any

from ..http import PoliteClient, xml_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source

NS = {"tt": "https://teamtailor.com/locations"}
PER_PAGE = 100
MAX_PAGES = 10
# remoteStatus values: none | hybrid | temporary | fully
WORKPLACE = {"hybrid": "Hybrid", "temporary": "Temporarily remote", "fully": "Remote"}


@register
class TeamtailorSource(Source):
    kind = "teamtailor"

    def __init__(self, site: str, company: str, default_location: str | None = None):
        site = site.removeprefix("https://").removeprefix("http://").strip("/")
        if "." not in site:  # bare slug
            site = f"{site}.teamtailor.com"
        super().__init__(name=site.removesuffix(".teamtailor.com"), company=company)
        self.base = f"https://{site}"
        self.default_location = default_location

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> TeamtailorSource:
        return cls(site=entry["site"], company=entry.get("name") or entry["site"],
                   default_location=entry.get("default_location"))

    async def fetch(self, http: PoliteClient) -> list[Job]:
        jobs: list[Job] = []
        for page in range(MAX_PAGES):
            resp = await http.get(f"{self.base}/jobs.rss", params={"per_page": PER_PAGE, "offset": page * PER_PAGE})
            batch = xml_body(resp, lambda text: parse_rss(text, self.feed, self.company, self.default_location))
            jobs += batch
            if len(batch) < PER_PAGE:
                break
        return jobs


def _text(el: ET.Element, path: str) -> str:
    found = el.find(path, NS)
    return (found.text or "").strip() if found is not None and found.text else ""


def _location(loc: ET.Element) -> str:
    name = _text(loc, "tt:name")
    city, country = _text(loc, "tt:city"), _text(loc, "tt:country")
    # Keep the name ("Romania (remote)" carries the remote flag); add the country when the name lacks it.
    if name and country and country.lower() not in name.lower():
        return f"{name}, {country}"
    return name or ", ".join(p for p in (city, country) if p)


def parse_rss(xml_text: str, feed: str, company: str, default_location: str | None = None) -> list[Job]:
    """`default_location` (from config) is used for jobs without any location, e.g. a Romanian company's
    postings that name no city."""
    root = ET.fromstring(xml_text)
    jobs = []
    for item in root.iter("item"):
        locations = [s for s in (_location(loc) for loc in item.iterfind("tt:locations/tt:location", NS)) if s]
        locations = locations or ([default_location] if default_location else [])
        status = _text(item, "remoteStatus")
        pub = _text(item, "pubDate")
        extra = {k: v for k, v in (("Department", _text(item, "tt:department")), ("Role", _text(item, "tt:role")),
                                   ("Workplace", WORKPLACE.get(status, ""))) if v}
        jobs.append(Job(
            source="teamtailor",
            job_id=_text(item, "guid") or _text(item, "link"),
            feed=feed,
            title=_text(item, "title"),
            company=company,
            location="; ".join(locations),
            url=_text(item, "link"),
            # XML parsing already unescaped the entity-escaped HTML once.
            description=html_to_text(_text(item, "description")),
            posted_at=parsedate_to_datetime(pub) if pub else None,
            remote=True if status == "fully" else None,
            extra=extra,
        ))
    return jobs
