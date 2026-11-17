"""Workable public jobs widget: https://help.workable.com/hc/en-us/articles/115012801727

GET apply.workable.com/api/v1/widget/accounts/{account}?details=true returns every published job with its
description, so no detail requests are needed. The widget lists a job once per location, so entries are merged
by shortcode. Dates are day-only, so like Workday the date goes in extra["Posted date"] and posted_at stays None.
"""
from __future__ import annotations

from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source

BASE = "https://apply.workable.com/api/v1/widget/accounts"


@register
class WorkableSource(Source):
    kind = "workable"

    def __init__(self, account: str, company: str):
        super().__init__(name=account, company=company)
        self.account = account

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> WorkableSource:
        return cls(account=entry["account"], company=entry.get("name") or entry["account"])

    async def fetch(self, http: PoliteClient) -> list[Job]:
        resp = await http.get(f"{BASE}/{self.account}", params={"details": "true"})
        return parse_widget(json_body(resp), self.feed, self.company)


def _place(loc: dict[str, Any]) -> str:
    return ", ".join(p for p in (loc.get("city"), loc.get("region"), loc.get("country")) if p)


def _places(j: dict[str, Any]) -> list[str]:
    locs = [_place(loc) for loc in j.get("locations") or [] if not loc.get("hidden")]
    return [p for p in locs if p] or [p for p in [_place({"city": j.get("city"), "region": j.get("state"),
                                                          "country": j.get("country")})] if p]


def parse_widget(payload: dict[str, Any], feed: str, company: str) -> list[Job]:
    jobs: dict[str, Job] = {}
    for j in payload.get("jobs") or []:
        places = _places(j)
        if (job := jobs.get(j["shortcode"])) is not None:  # same job, another location
            known = job.location.split("; ")
            job.location = "; ".join(known + [p for p in places if p not in known])
            job.remote = job.remote or (True if j.get("telecommuting") else None)
            continue
        remote = bool(j.get("telecommuting"))
        extra = {k: str(v) for k, v in (("Department", j.get("department")),
                                        ("Employment type", j.get("employment_type")),
                                        ("Experience", j.get("experience")),
                                        ("Education", j.get("education")),
                                        ("Function", j.get("function")),
                                        ("Workplace", "Remote" if remote else None),
                                        ("Posted date", j.get("published_on"))) if v}
        jobs[j["shortcode"]] = Job(
            source="workable",
            job_id=j["shortcode"],
            feed=feed,
            title=j["title"].strip(),
            company=company,
            location="; ".join(places),
            url=j.get("url") or j.get("shortlink") or "",
            description=html_to_text(j.get("description") or ""),
            posted_at=None,
            remote=True if remote else None,
            extra=extra,
        )
    return list(jobs.values())
