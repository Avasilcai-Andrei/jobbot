"""Stage 2 scoring: a common Scorer interface with prompt building, truncation, validation and caching.

Backends only implement `_complete()` (and optionally `check_available()`).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import date
from typing import ClassVar

from pydantic import BaseModel, ValidationError, field_validator

from ..config import Config
from ..db import DB
from ..models import Job
from ..text import fold, norm_company

log = logging.getLogger(__name__)

PROMPT_VERSION = "8"

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reason": {"type": "string"},
        "missing_requirements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "reason", "missing_requirements"],
    "additionalProperties": False,
}

# No tokenizer for local models, so estimate. Mixed English/Romanian text with markup leftovers
# averages about 3.5 characters per token; estimating high keeps the prompt inside the window.
CHARS_PER_TOKEN = 3.5
TRUNCATION_NOTE = "\n[... description truncated to fit the model's context window]"
EMPLOYER_PLACEHOLDER = "[Company]"
# Prepended to the system prompt for a second opinion. Ollama reuses the KV cache of the longest matching prompt
# prefix, and how much it reuses (which depends on the previous job) shifts borderline scores. A different first
# line makes it evaluate the whole prompt again (measured: 2592 tokens evaluated vs 4 for a repeat; a leading
# newline alone is stripped and still hits the cache).
SECOND_REVIEW_PREFIX = "(Second review.)\n"


class MatchResult(BaseModel):
    score: int
    reason: str
    missing_requirements: list[str]
    scores: list[int] | None = None  # set by score_stable when the job was scored twice; `score` is their mean

    @field_validator("score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"score {v} outside 0-100")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty reason")
        return v.strip()


class ScoreUnavailable(Exception):
    """The backend can't be reached / isn't set up. Skip scoring for the rest of the run."""


class ScoreUnreachable(ScoreUnavailable):
    """The server did not answer at all (not running yet, restarting). Unlike a missing model or key this
    usually heals by itself, so the pipeline holds the jobs instead of sending them [unscored] at once."""


class ScoreFailed(Exception):
    """This one job couldn't be scored (e.g. malformed output twice)."""


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def truncate_to_tokens(text: str, budget_tokens: int) -> tuple[str, bool]:
    """Cut text to roughly `budget_tokens`, preferring a paragraph or sentence boundary."""
    limit = int(budget_tokens * CHARS_PER_TOKEN) - len(TRUNCATION_NOTE)
    if len(text) <= limit + len(TRUNCATION_NOTE):
        return text, False
    cut = text[: max(limit, 0)]
    for sep in ("\n\n", "\n", ". "):
        pos = cut.rfind(sep)
        if pos >= limit * 0.8:
            cut = cut[: pos + (1 if sep == ". " else 0)]
            break
    return cut.rstrip() + TRUNCATION_NOTE, True


def build_system_prompt(cfg: Config, cv_text: str) -> str:
    roles = ", ".join(r.name for r in cfg.profile.roles)
    locations = "; ".join(
        f"{r.name}{' (remote only)' if r.remote_only else ''}{' - lower priority' if r.priority == 'low' else ''}"
        for r in cfg.locations.rules if r.enabled
    )
    years = cfg.profile.exclude_if_years_at_least
    return f"""You screen job postings for one early-career candidate and score how well each posting fits them.

<candidate_cv>
{cv_text}
</candidate_cv>

<preferences>
Target roles: {roles}. The candidate wants junior, associate, graduate or entry-level positions.
Locations: {locations}.
Hard exclusions: senior/lead/principal/manager roles{f", roles requiring {years}+ years of experience" if years else ""}.
</preferences>

Score 0-100:
- 85-100: a target role at junior level, and the CV covers most must-have requirements.
- 65-84: a good fit worth applying to; the gaps are learnable.
- 40-64: an adjacent role, or significant gaps.
- 0-39: poor fit, or the posting hits a hard exclusion.
Judge seniority from the stated requirements, not the title alone.

Answer with a JSON object with exactly these keys:
- "score": integer 0-100
- "reason": 2-3 short sentences to the candidate explaining the score
- "missing_requirements": list of concrete must-have requirements from the posting that the CV does not show, at most 6; empty list if none"""


def cv_employer_pattern(company: str, cv_folded: str) -> re.Pattern[str] | None:
    """A pattern for the job's employer name if the (folded) CV mentions it, else None.

    Small models score a posting higher just because the CV lists the same employer (working-student roles in
    unrelated fields got 95), and telling them to ignore it didn't work, so the name is hidden instead."""
    name = norm_company(company)
    if len(name) < 3 or not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", cv_folded):
        return None
    body = r"\s+".join(re.escape(w) for w in name.split())
    # Long names also match inside words ("LikeAContoso"); short ones ("ING") only as whole words.
    return re.compile(body if len(name) >= 5 else r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def hide_cv_employer(job: Job, cv_folded: str) -> Job:
    """A copy of the job with its employer's name replaced by EMPLOYER_PLACEHOLDER, if the CV names it."""
    rx = cv_employer_pattern(job.company, cv_folded)
    if rx is None:
        return job

    def hide(v: object) -> object:
        return rx.sub(EMPLOYER_PLACEHOLDER, v) if isinstance(v, str) else v

    return replace(job, title=hide(job.title), company=EMPLOYER_PLACEHOLDER, description=hide(job.description),
                   extra={k: hide(v) for k, v in job.extra.items()})


def format_job(job: Job, location_note: str | None, description: str, today: date | None = None) -> str:
    lines = [
        f"Today's date: {(today or date.today()):%B %Y}",
        f"Title: {job.title}",
        f"Company: {job.company}",
        f"Location: {job.location}" + (f" ({location_note})" if location_note else ""),
    ]
    lines += [f"{k}: {v}" for k, v in job.extra.items() if not k.startswith("_")]
    return "<job>\n" + "\n".join(lines) + f"\n\nDescription:\n{description or '(none)'}\n</job>"


def parse_result(text: str) -> MatchResult:
    """Raises ValueError on anything that isn't a valid result object."""
    text = text.strip()
    if text.startswith("```"):  # tolerate a fenced block from models that ignore the format hint
        text = text.strip("`").removeprefix("json").strip()
    try:
        return MatchResult.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(str(e).splitlines()[0]) from e


class Scorer(ABC):
    backend: ClassVar[str]
    unreachable: bool = False          # set by check_available(): the last problem was "no answer at all"
    uses_gpu: ClassVar[bool] = False   # a local model that takes VRAM (the pipeline checks free VRAM before loading it)

    def __init__(self, cfg: Config, cv_text: str, db: DB, *, model: str, context_tokens: int,
                 max_concurrency: int):
        self.model = model
        self.db = db
        self.system = build_system_prompt(cfg, cv_text)
        self.cv_folded = fold(cv_text)
        self.context_tokens = context_tokens
        self.reserve_output_tokens = cfg.llm.reserve_output_tokens
        self.max_description_tokens = cfg.llm.max_description_tokens
        self._sem = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model}"

    async def check_available(self) -> str | None:
        """None if ready, else a one-line explanation (logged once per run)."""
        return None

    async def aclose(self) -> None:
        return None

    async def unload(self) -> None:
        """Free the model's memory now, if the backend holds any."""

    @abstractmethod
    async def _complete(self, system: str, messages: list[dict[str, str]]) -> str:
        """Return the model's raw text answer. Raise ScoreUnavailable / ScoreFailed on errors."""

    def description_budget(self, job: Job, location_note: str | None) -> int:
        fixed = approx_tokens(self.system) + approx_tokens(format_job(job, location_note, "")) + 32
        room = self.context_tokens - self.reserve_output_tokens - fixed
        if room < 300:
            log.warning("CV + instructions leave only %d tokens for the job in a %d-token context; "
                        "raise the context size", room, self.context_tokens)
        return max(min(self.max_description_tokens, room), 100)

    def prepare(self, job: Job, location_note: str | None = None) -> str:
        job = hide_cv_employer(job, self.cv_folded)
        description, truncated = truncate_to_tokens(job.description or "", self.description_budget(job, location_note))
        if truncated:
            log.info("Truncated description of %r to ~%d tokens", job.title, self.description_budget(job, location_note))
        return format_job(job, location_note, description)

    def cache_key(self, job_text: str, system: str | None = None) -> str:
        h = hashlib.sha256()
        for part in (PROMPT_VERSION, self.name, system or self.system, job_text):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    async def score_stable(self, job: Job, location_note: str | None, threshold: int,
                           margin: int | None) -> MatchResult:
        """Score; if the score is within `margin` of `threshold`, score again cold and use the (rounded) mean."""
        first = await self.score(job, location_note)
        if margin is None or abs(first.score - threshold) > margin:
            return first
        second = await self.score(job, location_note, second_opinion=True)
        mean = int((first.score + second.score) / 2 + 0.5)
        log.info("Scored %r twice near the threshold %d: %d and %d -> %d", job.title, threshold,
                 first.score, second.score, mean)
        # Keep the explanation that agrees with the final decision (same side of the threshold).
        pick = second if (second.score >= threshold) == (mean >= threshold) != (first.score >= threshold) else first
        return pick.model_copy(update={"score": mean, "scores": [first.score, second.score]})

    async def score(self, job: Job, location_note: str | None = None, *,
                    second_opinion: bool = False) -> MatchResult:
        job_text = self.prepare(job, location_note)
        system = SECOND_REVIEW_PREFIX + self.system if second_opinion else self.system
        key = self.cache_key(job_text, system)
        if (cached := self.db.llm_cache_get(key)) is not None:
            return MatchResult.model_validate(cached)

        messages = [{"role": "user", "content": job_text}]
        async with self._sem:
            text = await self._complete(system, messages)
            try:
                result = parse_result(text)
            except ValueError as first:
                log.warning("%s gave malformed output for %r (%s); retrying once", self.name, job.title, first)
                messages += [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content": f"That was not valid: {first}. Reply with only the JSON object "
                                                f"with keys score (integer 0-100), reason and missing_requirements."},
                ]
                text = await self._complete(system, messages)
                try:
                    result = parse_result(text)
                except ValueError as second:
                    raise ScoreFailed(f"malformed output twice ({second})") from None

        self.db.llm_cache_put(key, self.name, result.model_dump())
        return result
