"""Workday public career-site API (the "cxs" JSON endpoints behind *.myworkdayjobs.com)."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import fold, html_to_text
from . import register
from .base import Source

log = logging.getLogger(__name__)

PAGE = 20          # Workday rejects larger pages
MAX_PAGES = 15
FACET_TTL = 24 * 3600
_MULTI_LOCATION = re.compile(r"^\d+ Locations$")
# Used when a tenant has no country facet, only a list of office locations.
DEFAULT_LOCATION_KEYWORDS = ["romania", "rou", "cluj", "bucharest", "bucuresti", "iasi", "timisoara", "brasov"]


@register
class WorkdaySource(Source):
    kind = "workday"

    def __init__(self, tenant: str, wd: int | None, site: str, company: str, country: str | None = "Romania",
                 location_keywords: list[str] | None = None):
        super().__init__(name=f"{tenant}/{site}", company=company)
        self.tenant, self.site, self.country = tenant, site, country
        self.location_keywords = location_keywords or DEFAULT_LOCATION_KEYWORDS
        # wd=None: the tenant has moved to a host without a data-centre number (e.g. NXP).
        self.host = f"https://{tenant}.wd{wd}.myworkdayjobs.com" if wd else f"https://{tenant}.myworkdayjobs.com"
        self.api = f"{self.host}/wday/cxs/{tenant}/{site}"
        self._facets: dict[str, list[str]] | None = None
        self._facets_at = 0.0

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> WorkdaySource:
        return cls(tenant=entry["tenant"], wd=int(entry["wd"]) if entry.get("wd") else None, site=entry["site"],
                   company=entry.get("name") or entry["tenant"], country=entry.get("country", "Romania"),
                   location_keywords=entry.get("location_keywords"))

    async def _country_facets(self, http: PoliteClient) -> dict[str, list[str]]:
        """Find the tenant's facet ids for the country (ids differ per tenant). Cached for a day."""
        if not self.country:
            return {}
        if self._facets is not None and time.monotonic() - self._facets_at < FACET_TTL:
            return self._facets
        resp = await http.post(f"{self.api}/jobs", json={"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""})
        facets = json_body(resp).get("facets") or []
        found = find_country_facet(facets, self.country)
        if found is not None:
            applied = {found[0]: [found[1]]}
        else:
            ids = find_location_values(facets, self.location_keywords)
            if not ids:
                raise ValueError(f"{self.feed}: no {self.country!r} country facet and no location matching "
                                 f"{self.location_keywords}; set country: null to fetch every job")
            applied = {"locations": ids}
        self._facets, self._facets_at = applied, time.monotonic()
        return applied

    async def fetch(self, http: PoliteClient) -> list[Job]:
        facets = await self._country_facets(http)
        jobs: list[Job] = []
        for page in range(MAX_PAGES):
            body = {"appliedFacets": facets, "limit": PAGE, "offset": page * PAGE, "searchText": ""}
            data = json_body(await http.post(f"{self.api}/jobs", json=body))
            jobs += parse_jobs(data, self)
            total = data.get("total") or 0
            if (page + 1) * PAGE >= total or not data.get("jobPostings"):
                break
        else:
            log.warning("%s: stopped after %d pages", self.feed, MAX_PAGES)
        return jobs

    async def fetch_details(self, http: PoliteClient, job: Job) -> Job:
        resp = await http.get(f"{self.api}{job.job_id}")
        return apply_details(job, json_body(resp))


def find_country_facet(facets: list[dict[str, Any]], country: str) -> tuple[str, str] | None:
    """Return (facetParameter, id) for the country value, searching nested facet groups."""
    for f in facets:
        param = f.get("facetParameter") or ""
        for v in f.get("values") or []:
            if "facetParameter" in v:  # nested group, e.g. locationMainGroup -> locationCountry
                if (hit := find_country_facet([v], country)) is not None:
                    return hit
            elif "country" in param.lower() and (v.get("descriptor") or "").lower() == country.lower():
                return param, v["id"]
    return None


def find_location_values(facets: list[dict[str, Any]], keywords: list[str]) -> list[str]:
    """Ids of `locations` facet values whose name contains one of the keywords (whole words)."""
    pats = [re.compile(r"(?<![a-z])" + re.escape(fold(k)) + r"(?![a-z])") for k in keywords]
    ids: list[str] = []
    for f in facets:
        for v in f.get("values") or []:
            if "facetParameter" in v:
                ids += find_location_values([v], keywords)
            elif f.get("facetParameter") == "locations" and any(p.search(fold(v.get("descriptor") or "")) for p in pats):
                ids.append(v["id"])
    return ids


def parse_jobs(payload: dict[str, Any], src: WorkdaySource) -> list[Job]:
    jobs = []
    for p in payload.get("jobPostings") or []:
        if not p.get("externalPath") or not p.get("title"):
            # Some tenants (seen on ING) return a stub entry {"bulletFields": ["REQ-..."]} with nothing else.
            log.debug("Skipping incomplete %s posting: %s", src.feed, str(p)[:120])
            continue
        location = p.get("locationsText") or ""
        extra = {}
        if _MULTI_LOCATION.match(location):
            extra["_location_incomplete"] = "1"  # the real list only comes with the details
        jobs.append(Job(
            source="workday",
            job_id=p["externalPath"],  # stable and unique per posting; also the details path
            feed=src.feed,
            title=p["title"].strip(),
            company=src.company,
            location=location,
            url=f"{src.host}/{src.site}{p['externalPath']}",
            posted_at=None,  # the list only says "Posted Today"; details have a date, not a time
            extra=extra,
        ))
    return jobs


def apply_details(job: Job, payload: dict[str, Any]) -> Job:
    info = payload.get("jobPostingInfo") or {}
    job.description = html_to_text(info.get("jobDescription") or "")
    locations = [info.get("location"), *(info.get("additionalLocations") or [])]
    if any(locations):
        job.location = "; ".join(filter(None, locations))
    job.extra.pop("_location_incomplete", None)
    if info.get("remoteType"):
        job.extra["Workplace"] = info["remoteType"]
        if "remote" in info["remoteType"].lower() and "hybrid" not in info["remoteType"].lower():
            job.remote = True
    if info.get("timeType"):
        job.extra["Time type"] = info["timeType"]
    if info.get("startDate"):
        job.extra["Posted date"] = info["startDate"]
    if info.get("externalUrl"):
        job.url = info["externalUrl"]
    return job
