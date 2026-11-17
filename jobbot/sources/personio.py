"""Personio public XML job feed: https://developer.personio.de/docs/retrieving-open-job-positions

GET https://{company}.jobs.personio.{com|de}/xml?language=en lists every open position with its description
sections, so no detail requests are needed. Offices are free-text names chosen by the company (usually a city).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

from ..http import PoliteClient, xml_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source

YEARS = {"lt-1": "less than 1 year", "1-2": "1-2 years", "2-5": "2-5 years", "5-7": "5-7 years",
         "7-10": "7-10 years", "10-15": "10-15 years", "gt-15": "more than 15 years"}


@register
class PersonioSource(Source):
    kind = "personio"

    def __init__(self, slug: str, company: str, domain: str = "com"):
        super().__init__(name=slug, company=company)
        self.base = f"https://{slug}.jobs.personio.{domain}"

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> PersonioSource:
        return cls(slug=entry["slug"], company=entry.get("name") or entry["slug"], domain=entry.get("domain", "com"))

    async def fetch(self, http: PoliteClient) -> list[Job]:
        resp = await http.get(f"{self.base}/xml", params={"language": "en"})
        jobs = xml_body(resp, lambda text: parse_xml(text, self.feed, self.company, self.base))
        if any(not j.description for j in jobs):
            # ?language=en leaves descriptions empty for jobs published only in another language (measured on
            # a German-only board); the default feed has them in the language they were written in.
            resp = await http.get(f"{self.base}/xml")
            native = {j.job_id: j.description
                      for j in xml_body(resp, lambda text: parse_xml(text, self.feed, self.company, self.base))}
            for j in jobs:
                j.description = j.description or native.get(j.job_id) or ""
        return jobs


def _text(el: ET.Element, tag: str) -> str:
    found = el.find(tag)
    return (found.text or "").strip() if found is not None else ""


def parse_xml(xml_text: str, feed: str, company: str, base: str) -> list[Job]:
    jobs = []
    for p in ET.fromstring(xml_text).iter("position"):
        offices = [o for o in [_text(p, "office"), *((o.text or "").strip() for o in p.iterfind("additionalOffices/office"))]
                   if o]
        sections = []
        for d in p.iterfind("jobDescriptions/jobDescription"):
            body = html_to_text(_text(d, "value"))
            if body:
                sections.append(f"{_text(d, 'name')}\n{body}".strip())
        created = _text(p, "createdAt")
        extra = {k: v for k, v in (("Department", _text(p, "department")),
                                   ("Employment type", _text(p, "employmentType")),
                                   ("Seniority", _text(p, "seniority")),
                                   ("Schedule", _text(p, "schedule")),
                                   ("Years of experience", YEARS.get(_text(p, "yearsOfExperience"), "")),
                                   ("Occupation", _text(p, "occupationCategory"))) if v}
        job_id = _text(p, "id")
        jobs.append(Job(
            source="personio",
            job_id=job_id,
            feed=feed,
            title=_text(p, "name"),
            company=company,
            location="; ".join(dict.fromkeys(offices)),
            url=f"{base}/job/{job_id}?language=en",
            description="\n\n".join(sections),
            posted_at=datetime.fromisoformat(created) if created else None,
            remote=True if any("remote" in o.lower() for o in offices) else None,
            extra=extra,
        ))
    return jobs
