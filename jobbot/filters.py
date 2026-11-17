"""Stage 1: cheap rule filter (role keywords, location, seniority, experience, unpaid, hard requirements)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config, HardRequirements, LocationRule
from .models import Job
from .text import fold

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "unu": 1, "doi": 2, "trei": 3, "patru": 4, "cinci": 5}
_NUM = r"(\d{1,2}|" + "|".join(_WORD_NUMBERS) + r")"
# "3+ years", "3-5 years", "minimum 3 years", "3 ani" -- followed closely by "experience"/"experienta".
_YEARS_RE = re.compile(_NUM + r"\s*(?:\+|plus)?\s*(?:(?:-|–|to|la)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?|ani)\b")
_EXPERIENCE_RE = re.compile(r"experien|\bexp\b")
# Without "experience": "3+ years in software, data or AI engineering", "3 years of Python", "2 years working in
# data", "3 years as a data engineer", "3 ani in domeniu", "3 ani de lucru in IT", "3 ani ca programator".
_FIELD_LEAD_RE = re.compile(r"[’']?\s*(?:(?:of\s+)?(?:professional\s+)?(?:work(?:ing)?\s+)?(?:in|as)\s+(?:an?\s+)?"
                            r"|of\s+(?!the\b|our\b|its\b)|de\s+(?:lucru|munca)\b\s*(?:in\s+|ca\s+)?"
                            r"|in\s+|ca\s+)(?=\w)")
# ...but not when the years are about something else: "5 years in a row", "3 ani in urma", "1 year of contract".
_NOT_FIELD_RE = re.compile(r"a row|business|existence|operation|the market|romania|cluj|bucharest|bucuresti|our\b|"
                           r"this\b|the company|urma|contract|age\b|growth|history|istori|piata")
# Company history, not a requirement: "for 30+ years, we've...", "our team has 10 years in fintech". A bare "we"
# doesn't count: requirement lists start with "what we need from you".
_COMPANY_HISTORY_RE = re.compile(r"\b(?:our|has|have been|founded|since|celebrat\w*|compania|firma|noastra|nostru)\b"
                                 r"|\b(?:for|since|past|last|timp de|de)\s*$")
_OPTIONAL_RE = re.compile(r"nice to have|preferred|is a plus|bonus|ideally|constituie un avantaj")
MAX_PLAUSIBLE_YEARS = 15  # "over 20 years of experience" is the company bragging, not a requirement
_COUNTRY_ONLY_RE = re.compile(r"\W*(romania|ro)\W*")
ENTRY_LEVEL = "Entry level"  # the role reported for titles matched only by profile.entry_level_keywords


def _phrase_re(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(fold(phrase)) + r"(?!\w)")


def _any_phrase_re(phrases: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(fold(p)) for p in sorted(phrases, key=len, reverse=True)) or r"(?!x)x"
    return re.compile(r"(?<!\w)(?:" + alts + r")(?!\w)")


# Hard requirements are judged per clause, so "Fluent English. German is a plus" keeps the job.
_CLAUSE_END = re.compile(r"[\n;()•]|\.\s")
_SENTENCE_END = re.compile(r"[\n;]|\.\s")
_OPTIONAL_CLAUSE_RE = re.compile(r"nice to have|preferred|preferably|a plus|bonus|ideally|advantage|optional"
                                 r"|avantaj|un plus|de dorit")
_ENGLISH_ALTERNATIVE = r"(?:english|engleza|romanian|romana)"
_DAYS_RE = re.compile(r"\b(\d|one|two|three|four|five|o|una|doua|trei|patru|cinci)\s*(?:\+|or more)?\s*"
                      r"(?:(?:-|–|to|or|sau|la)\s*(?:\d|two|three|four|five|doua|trei|patru|cinci)\s*)?"
                      r"(?:days?|zile|zi)\b")
_DAY_NUMBERS = {**_WORD_NUMBERS, "o": 1, "una": 1, "doua": 2}
_PER_MONTH_RE = re.compile(r"\W*(?:a|per|each|every|pe|/)?\s*(?:month|year|quarter|luna|an)\b")
_ONSITE_RE = re.compile(r"office|on-?site|on site|in[- ]person|birou|sediu|la fata locului")
_ROMANIA_RE = re.compile(r"\b(?:romania|cluj|napoca|bucharest|bucuresti|iasi|timisoara|brasov|sibiu|oradea|"
                         r"craiova|constanta|targu mures)\b")


def _clause(text: str, start: int, end: int, boundary: re.Pattern[str] = _CLAUSE_END, limit: int = 200) -> str:
    """The clause around text[start:end], cut at the nearest boundaries (at most `limit` chars each way)."""
    lo, hi = max(0, start - limit), min(len(text), end + limit)
    cuts = [m.end() for m in boundary.finditer(text, lo, start)]
    m = boundary.search(text, end, hi)
    return text[cuts[-1] if cuts else lo: m.start() if m else hi]


def _quote(clause: str, width: int = 70) -> str:
    q = " ".join(clause.split()).strip(" -,:")
    return f'"{q[:width]}…"' if len(q) > width else f'"{q}"'


class HardRequirementFilter:
    """Requirements the candidate can't meet: foreign-language fluency, Master's/PhD only, on-site days abroad."""

    def __init__(self, hr: HardRequirements):
        self.languages = [(name, _any_phrase_re(words)) for name, words in hr.languages.items() if words]
        self.required_words = _any_phrase_re(hr.language_required_words)
        self.alternatives = {
            name: re.compile(rf"{_ENGLISH_ALTERNATIVE}\s+(?:or|sau|and/or|/)\s+(?:{rx.pattern})"
                             rf"|(?:{rx.pattern})\s+(?:or|sau|and/or|/)\s+{_ENGLISH_ALTERNATIVE}")
            for name, rx in self.languages
        }
        self.graduate_only = [re.compile(rx) for rx in hr.graduate_only_regex]
        self.undergraduate = [re.compile(rx) for rx in hr.undergraduate_regex]
        self.min_days_abroad = hr.onsite_abroad.min_days
        self.abroad = _any_phrase_re(hr.onsite_abroad.places)

    def language(self, text: str) -> str | None:
        for name, rx in self.languages:
            for m in rx.finditer(text):
                clause = _clause(text, m.start(), m.end())
                if not self.required_words.search(clause) or _OPTIONAL_CLAUSE_RE.search(clause):
                    continue
                if self.alternatives[name].search(clause):  # "English or German" doesn't require German
                    continue
                return f"requires {name}: {_quote(clause)}"
        return None

    def degree(self, text: str) -> str | None:
        if any(rx.search(text) for rx in self.undergraduate):
            return None
        for rx in self.graduate_only:
            for m in rx.finditer(text):
                clause = _clause(text, m.start(), m.end())
                if _OPTIONAL_CLAUSE_RE.search(clause) or "equivalent" in clause:
                    continue
                return f"Master's/PhD only: {_quote(clause)}"
        return None

    def onsite_abroad(self, text: str) -> str | None:
        if self.min_days_abroad is None:
            return None
        for m in _DAYS_RE.finditer(text):
            days = int(m.group(1)) if m.group(1).isdigit() else _DAY_NUMBERS[m.group(1)]
            if days < self.min_days_abroad or _PER_MONTH_RE.match(text, m.end()):
                continue
            sentence = _clause(text, m.start(), m.end(), _SENTENCE_END)
            if _ONSITE_RE.search(sentence) and self.abroad.search(sentence) and not _ROMANIA_RE.search(sentence):
                return f"{days}+ on-site days outside Romania: {_quote(sentence)}"
        return None

    def check(self, text: str) -> str | None:
        """The first unmet hard requirement in the folded title + description, or None."""
        return self.language(text) or self.degree(text) or self.onsite_abroad(text)


@dataclass
class Verdict:
    ok: bool
    reason: str
    role: str | None = None
    location_rule: LocationRule | None = None

    @property
    def priority(self) -> str | None:
        return self.location_rule.priority if self.location_rule else None


def _years_in_field(m: re.Match[str], text: str) -> bool:
    """True for "N years in/of/as <field>" (and Romanian "N ani in/de lucru in/ca <field>") as a requirement."""
    lead = _FIELD_LEAD_RE.match(text, m.end())
    if not lead or _NOT_FIELD_RE.match(text, lead.end()):
        return False
    return not _COMPANY_HISTORY_RE.search(text[max(0, m.start() - 30): m.start()])


def required_years(text: str) -> int | None:
    """Largest "N years of experience" requirement found in the (folded) text, if any."""
    best: int | None = None
    for m in _YEARS_RE.finditer(text):
        raw = m.group(1)
        n = int(raw) if raw.isdigit() else _WORD_NUMBERS[raw]
        if n > MAX_PLAUSIBLE_YEARS:
            continue
        after = text[m.end(): m.end() + 60]
        before = text[max(0, m.start() - 60): m.start()]
        if not (_EXPERIENCE_RE.search(after) or _EXPERIENCE_RE.search(before[-25:]) or _years_in_field(m, text)):
            continue
        if _OPTIONAL_RE.search(before) or _OPTIONAL_RE.search(after[:30]):
            continue
        best = n if best is None else max(best, n)
    return best


class RuleFilter:
    def __init__(self, cfg: Config):
        p = cfg.profile
        self.roles = [(r.name, [_phrase_re(k) for k in r.keywords]) for r in p.roles]
        if p.entry_level_keywords:  # last, so a title naming a target role keeps that role
            self.roles.append((ENTRY_LEVEL, [_phrase_re(k) for k in p.entry_level_keywords]))
        self.entry_phrases = [_phrase_re(k) for k in p.entry_level_keywords]
        self.email_needs_entry_level = cfg.notify.digest.require_entry_level_title and bool(self.entry_phrases)
        self.exclude_words = [(w, _phrase_re(w)) for w in p.exclude_title_words]
        self.exclude_at_employer = [(x.employer, _phrase_re(x.employer), [(w, _phrase_re(w)) for w in x.title_contains])
                                    for x in p.exclude_title_at_employer]
        self.min_years_excluded = p.exclude_if_years_at_least
        self.exclude_text = [re.compile(rx) for rx in p.exclude_text_regex]
        self.hard = HardRequirementFilter(p.hard_requirements)
        self.remote_words = [_phrase_re(w) for w in cfg.locations.remote_keywords]
        # Job-alert emails often say just "Romania" and carry no description to show whether it is remote or where.
        # Such an email job is kept (the digest marks it) rather than dropped; ATS jobs still need a real match.
        self.country_only_rule = LocationRule(name="Romania (country only)", match=["romania"], priority="low")
        self.location_rules = [
            (rule, [_phrase_re(m) for m in rule.match]) for rule in cfg.locations.rules if rule.enabled
        ]

    def _is_remote(self, job: Job, loc: str) -> bool:
        if job.remote:
            return True
        haystack = " ".join([loc, fold(job.title), *(fold(v) for v in job.extra.values())])
        if any(rx.search(haystack) for rx in self.remote_words):
            return True
        # Country-only location ("Romania"): the description is the only place that says remote.
        # Before it is fetched, give the job the benefit of the doubt; check() decides later.
        if re.fullmatch(r"\W*(romania|ro)\W*", loc):
            if job.description is None:
                return True
            return any(rx.search(fold(job.description)) for rx in self.remote_words)
        return False

    def match_location(self, job: Job) -> LocationRule | None:
        if job.extra.get("_location_incomplete") and job.description is None:
            # e.g. Workday's "2 Locations": the real list arrives with the details, so let it through for now.
            return next((rule for rule, _ in self.location_rules), None)
        loc = fold(job.location)
        for rule, patterns in self.location_rules:
            if not any(rx.search(loc) for rx in patterns):
                continue
            if rule.remote_only and not self._is_remote(job, loc):
                continue
            return rule
        if job.extra.get("_from_email") and _COUNTRY_ONLY_RE.fullmatch(loc):
            return self.country_only_rule
        return None

    def prefilter(self, job: Job) -> Verdict:
        """Title + location only, so it runs before we spend a request on the description."""
        title = fold(job.title)
        for word, rx in self.exclude_words:
            if rx.search(title):
                return Verdict(False, f"title has '{word}'")
        company = fold(job.company)
        for employer, employer_rx, words in self.exclude_at_employer:
            if employer_rx.search(company):
                for word, rx in words:
                    if rx.search(title):
                        return Verdict(False, f"title has '{word}' at {employer}")
        role = next((name for name, pats in self.roles if any(rx.search(title) for rx in pats)), None)
        if role is None:
            return Verdict(False, "title matches no target role")
        if (self.email_needs_entry_level and job.extra.get("_from_email")
                and not any(rx.search(title) for rx in self.entry_phrases)):
            # An email has no description to show the seniority or the years asked for, so only an explicit
            # entry-level word in the title gets such a job into the digest.
            return Verdict(False, "email: no entry-level signal", role=role)
        rule = self.match_location(job)
        if rule is None:
            return Verdict(False, f"location '{job.location}' not targeted", role=role)
        return Verdict(True, "ok", role=role, location_rule=rule)

    def check(self, job: Job) -> Verdict:
        """Full stage-1 check; call after the description has been fetched."""
        v = self.prefilter(job)
        if not v.ok:
            return v
        text = fold(f"{job.title}\n{job.description or ''}")
        for rx in self.exclude_text:
            if rx.search(text):
                return Verdict(False, f"matches exclusion /{rx.pattern}/", v.role, v.location_rule)
        if self.min_years_excluded is not None:
            years = required_years(text)
            if years is not None and years >= self.min_years_excluded:
                return Verdict(False, f"asks for {years}+ years of experience", v.role, v.location_rule)
        if (unmet := self.hard.check(text)):
            return Verdict(False, unmet, v.role, v.location_rule)
        # Re-run location: a country-only location may now be confirmed remote by the description.
        rule = self.match_location(job)
        if rule is None:
            return Verdict(False, f"location '{job.location}' not targeted", v.role)
        return Verdict(True, "ok", v.role, rule)
