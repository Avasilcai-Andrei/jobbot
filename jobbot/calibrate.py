"""`calibrate`: score labelled jobs with one or more models and suggest a notify threshold.

calibration/labels.yaml   - which jobs, and your label for each (yes | maybe | no)
calibration/jobs/<id>.json - saved job snapshots (title, company, location, description, ...)
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .cv import load_cv_text
from .db import DB, row_to_job
from .filters import RuleFilter
from .http import PoliteClient
from .models import Job
from .scoring import MatchResult, ScoreFailed, ScoreUnavailable, make_scorer
from .scoring.base import approx_tokens
from .sources import REGISTRY, build_sources
from .sources.base import Source

log = logging.getLogger(__name__)

LABELS = ("yes", "maybe", "no")


@dataclass
class LabelledJob:
    id: str
    label: str
    job: Job
    note: str = ""


@dataclass
class Outcome:
    score: int | None
    seconds: float
    error: str | None = None
    prompt_tokens: int | None = None
    estimated_tokens: int | None = None
    result: MatchResult | None = None


@dataclass
class ModelRun:
    name: str
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    unavailable: str | None = None


# --- files ----------------------------------------------------------------------

def calibration_dir(cfg: Config) -> Path:
    return cfg.resolve(Path("calibration"))


def job_to_snapshot(job: Job, **meta: Any) -> dict[str, Any]:
    return {
        "source": job.source, "feed": job.feed, "job_id": job.job_id, "title": job.title, "company": job.company,
        "location": job.location, "url": job.url, "description": job.description, "remote": job.remote,
        "posted_at": job.posted_at.isoformat() if job.posted_at else None,
        "extra": {k: v for k, v in job.extra.items() if not k.startswith("_")},
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **meta,
    }


def snapshot_to_job(d: dict[str, Any]) -> Job:
    return Job(source=d["source"], job_id=d["job_id"], feed=d["feed"], title=d["title"], company=d["company"],
               location=d["location"], url=d["url"], description=d.get("description") or "",
               posted_at=datetime.fromisoformat(d["posted_at"]) if d.get("posted_at") else None,
               remote=d.get("remote"), extra=d.get("extra") or {})


def load_labelled(cal_dir: Path) -> list[LabelledJob]:
    path = cal_dir / "labels.yaml"
    labels = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    out = []
    for entry in labels.get("jobs") or []:
        raw = entry.get("label", "")
        # YAML 1.1 reads bare yes/no as booleans
        label = {True: "yes", False: "no"}.get(raw, str(raw)).strip().lower() if isinstance(raw, bool) else str(raw).strip().lower()
        if label not in LABELS:
            log.warning("Skipping %s: label %r is not one of %s", entry.get("id"), entry.get("label"), LABELS)
            continue
        snap = cal_dir / "jobs" / f"{entry['id']}.json"
        if not snap.exists():
            log.warning("Skipping %s: %s is missing", entry["id"], snap)
            continue
        out.append(LabelledJob(entry["id"], label, snapshot_to_job(json.loads(snap.read_text(encoding="utf-8"))),
                               entry.get("note") or ""))
    return out


def snapshot_id(job: Job) -> str:
    raw = f"{job.feed}-{job.job_id}".lower()
    return "".join(c if c.isalnum() else "-" for c in raw).strip("-")[:80]


def source_for_feed(cfg: Config, feed: str) -> Source:
    """A configured feed, or an ad-hoc greenhouse/lever/smartrecruiters board (e.g. greenhouse:databricks)."""
    for s in build_sources(cfg):
        if s.feed == feed:
            return s
    kind, _, name = feed.partition(":")
    key = {"greenhouse": "slug", "lever": "slug", "smartrecruiters": "id"}.get(kind)
    if key is None or kind not in REGISTRY:
        raise SystemExit(f"Feed {feed!r} is not in config.yaml (Workday feeds must be configured first)")
    return REGISTRY[kind].from_entry({key: name, "name": name})


async def add_job(cfg: Config, feed: str, job_id: str, label: str, note: str, location: str | None) -> Path:
    source = source_for_feed(cfg, feed)
    async with PoliteClient(cfg.http.user_agent) as http:
        jobs = await source.fetch(http)
        job = next((j for j in jobs if j.job_id == job_id), None)
        if job is None:
            raise SystemExit(f"Job {job_id!r} is not currently listed on {feed}")
        if job.description is None:
            await source.fetch_details(http, job)
    meta: dict[str, Any] = {}
    if location:  # judge the role itself, as if it were in one of your locations
        meta["original_location"] = job.location
        job.location = location
    cal = calibration_dir(cfg)
    (cal / "jobs").mkdir(parents=True, exist_ok=True)
    sid = snapshot_id(job)
    path = cal / "jobs" / f"{sid}.json"
    path.write_text(json.dumps(job_to_snapshot(job, **meta), ensure_ascii=False, indent=1), encoding="utf-8")
    labels = cal / "labels.yaml"
    if not labels.exists():
        labels.write_text("# label: yes | maybe | no\njobs:\n", encoding="utf-8")
    existing = yaml.safe_load(labels.read_text(encoding="utf-8")) or {}
    if not any(e.get("id") == sid for e in existing.get("jobs") or []):
        with labels.open("a", encoding="utf-8") as f:
            f.write(f"  - id: {sid}\n    label: {label}\n    note: {json.dumps(note or job.title, ensure_ascii=False)}\n")
    return path


def import_marks(cfg: Config, db: DB) -> list[tuple[str, str, str, str]]:
    """Copy good/bad alert marks into the calibration set (good = yes, bad = no), from the stored jobs
    (no network: the posting may be gone). Returns (snapshot id, label, title, outcome) per marked alert.
    A job already in labels.yaml is left alone, even when its label differs (outcome says so)."""
    cal = calibration_dir(cfg)
    (cal / "jobs").mkdir(parents=True, exist_ok=True)
    labels = cal / "labels.yaml"
    if not labels.exists():
        labels.write_text("# label: yes | maybe | no\njobs:\n", encoding="utf-8")
    existing = {e.get("id"): e for e in (yaml.safe_load(labels.read_text(encoding="utf-8")) or {}).get("jobs") or []}
    out: list[tuple[str, str, str, str]] = []
    for r in db.alerts_with_marks():
        if r["mark"] not in ("good", "bad"):
            continue
        job = row_to_job(r)
        label = "yes" if r["mark"] == "good" else "no"
        sid = snapshot_id(job)
        if sid in existing:
            old = str(existing[sid].get("label")).lower()
            old = {"true": "yes", "false": "no"}.get(old, old)
            out.append((sid, label, job.title, "already in the set" + ("" if old == label else f" as {old!r}: kept")))
            continue
        job.description = job.description or ""
        (cal / "jobs" / f"{sid}.json").write_text(
            json.dumps(job_to_snapshot(job, source_mark=r["mark"]), ensure_ascii=False, indent=1), encoding="utf-8")
        note = f"{r['mark_note'] or job.title} (own label, marked {r['mark']})"
        with labels.open("a", encoding="utf-8") as f:
            f.write(f"  - id: {sid}\n    label: {label}\n    note: {json.dumps(note, ensure_ascii=False)}\n")
        out.append((sid, label, job.title, "added"))
    return out


# --- scoring ----------------------------------------------------------------

async def score_with_model(cfg: Config, cv_text: str, items: list[LabelledJob], model: str | None) -> ModelRun:
    db = DB(":memory:")  # fresh cache: we want real timings and fresh answers
    scorer, why = make_scorer(cfg, db, model=model, cv_text=cv_text)
    if scorer is None:
        return ModelRun(name=model or cfg.llm.backend, unavailable=why)
    run = ModelRun(name=scorer.name)
    try:
        if (problem := await scorer.check_available()):
            run.unavailable = problem
            return run
        for i, item in enumerate(items, 1):
            prompt_estimate = approx_tokens(scorer.system) + approx_tokens(scorer.prepare(item.job, None))
            t0 = time.perf_counter()
            try:
                # Same as the pipeline: near the threshold, a second cold score and the mean.
                res = await scorer.score_stable(item.job, None, cfg.notify.min_score, cfg.llm.rescore_margin)
                out = Outcome(res.score, time.perf_counter() - t0, result=res)
            except ScoreUnavailable as e:
                run.unavailable = str(e)
                return run
            except ScoreFailed as e:
                out = Outcome(None, time.perf_counter() - t0, error=str(e))
            stats = getattr(scorer, "last_stats", {}) or {}
            out.prompt_tokens = stats.get("prompt_eval_count")
            out.estimated_tokens = prompt_estimate
            run.outcomes[item.id] = out
            log.info("[%s] %d/%d %-50s %s (%.1fs)", scorer.name, i, len(items), item.job.title[:50],
                     out.score if out.score is not None else f"ERROR {out.error}", out.seconds)
        return run
    finally:
        await scorer.aclose()
        db.close()


# --- analysis -----------------------------------------------------------------

def auc(pos: list[int], neg: list[int]) -> float | None:
    """Probability that a random 'yes' scores above a random 'no' (ties count half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def suggest_threshold(yes: list[int], no: list[int]) -> tuple[int, int, int] | None:
    """Maximise recall(yes) - false-positive-rate(no). Returns (suggested, low, high) of the best range."""
    if not yes or not no:
        return None
    best, best_ts = -2.0, []
    for t in range(0, 101):
        j = sum(s >= t for s in yes) / len(yes) - sum(s >= t for s in no) / len(no)
        if j > best + 1e-9:
            best, best_ts = j, [t]
        elif abs(j - best) <= 1e-9:
            best_ts.append(t)
    # The best thresholds form a range between the highest 'no' and lowest 'yes' scores.
    # Take the lower part of the range: a missed job costs more than an extra alert.
    lo, hi = min(best_ts), max(best_ts)
    return lo + (hi - lo) // 3, lo, hi


def _fmt_scores(vals: list[int]) -> str:
    if not vals:
        return "-"
    return f"n={len(vals):<2} mean {statistics.mean(vals):5.1f}  min {min(vals):3}  max {max(vals):3}"


def histogram(vals_by_label: dict[str, list[int]]) -> list[str]:
    lines = []
    for lo in range(0, 100, 10):
        hi = 100 if lo == 90 else lo + 9
        cells = []
        for label, mark in (("yes", "Y"), ("maybe", "?"), ("no", "N")):
            n = sum(lo <= v <= hi for v in vals_by_label.get(label, []))
            cells.append(mark * n)
        lines.append(f"    {lo:3}-{hi:<3} | {''.join(cells)}")
    return lines


def rule_verdicts(cfg: Config, items: list[LabelledJob]) -> dict[str, str | None]:
    """Stage-1 result per job: None if it passes the rules, else the reason it is dropped.
    Like the labels, this judges the role, not the place: each job is checked as if it were in the first
    location rule's area (so only the title and description rules can drop it)."""
    rules = RuleFilter(cfg)
    first = next(r for r in cfg.locations.rules if r.enabled)
    here = {i.id: replace(i.job, location=first.match[0], remote=True, extra=dict(i.job.extra)) for i in items}
    return {i.id: (None if (v := rules.check(here[i.id])).ok else v.reason) for i in items}


def report(items: list[LabelledJob], runs: list[ModelRun], current_threshold: int,
           dropped: dict[str, str | None] | None = None) -> str:
    """`dropped` (from rule_verdicts) adds a rules column and the end-to-end result: rules, then threshold."""
    dropped = dropped or {}
    out: list[str] = []
    ok_runs = [r for r in runs if not r.unavailable]
    for r in runs:
        if r.unavailable:
            out.append(f"!! {r.name}: not run - {r.unavailable}")
    if not ok_runs:
        return "\n".join(out)

    w = 46
    head = (f"{'job':{w}} {'label':6}" + "".join(f" {r.name.split(':', 1)[-1][:16]:>16}" for r in ok_runs)
            + ("  rules" if dropped else ""))
    out += ["", head, "-" * len(head)]
    order = {lab: i for i, lab in enumerate(LABELS)}
    for item in sorted(items, key=lambda i: (order[i.label], i.job.title)):
        row = f"{(item.job.title + ' @ ' + item.job.company)[:w]:{w}} {item.label:6}"
        for r in ok_runs:
            o = r.outcomes.get(item.id)
            row += f" {('ERR' if o is None or o.score is None else str(o.score)):>16}"
        if dropped:
            row += f"  {'drop: ' + why[:70] if (why := dropped.get(item.id)) else 'pass'}"
        out.append(row)

    for r in ok_runs:
        by: dict[str, list[int]] = {lab: [] for lab in LABELS}
        for item in items:
            o = r.outcomes.get(item.id)
            if o and o.score is not None:
                by[item.label].append(o.score)
        outcomes = list(r.outcomes.values())
        errors = [o for o in outcomes if o.score is None]
        secs = [o.seconds for o in outcomes]
        # Ollama reuses the cached system-prompt prefix, so only the first call reports the full prompt size.
        first = outcomes[0] if outcomes else None
        a = auc(by["yes"], by["no"])
        sug = suggest_threshold(by["yes"], by["no"])
        out += ["", f"== {r.name}",
                f"  yes   {_fmt_scores(by['yes'])}",
                f"  maybe {_fmt_scores(by['maybe'])}",
                f"  no    {_fmt_scores(by['no'])}",
                "  distribution (Y = yes, ? = maybe, N = no):", *histogram(by),
                f"  separation (AUC yes vs no): {a:.2f}" if a is not None else "  separation: need both yes and no labels",
                f"  failed/malformed: {len(errors)} of {len(outcomes)}"
                + (f"  ({'; '.join(sorted({e.error or '' for e in errors}))[:120]})" if errors else ""),
                f"  time per job: median {statistics.median(secs):.1f}s, max {max(secs):.1f}s" if secs else ""]
        twice = [(i, o) for i in items if (o := r.outcomes.get(i.id)) and o.result and o.result.scores]
        out.append(f"  scored twice near the threshold: {len(twice)} of {len(outcomes)}"
                   + "".join(f"\n    {i.label:5} {i.job.title[:46]:46} {o.result.scores[0]:3} and "
                             f"{o.result.scores[1]:3} -> {o.score}" for i, o in twice))
        if first and first.prompt_tokens and first.estimated_tokens:
            out.append(f"  prompt size (first job): {first.prompt_tokens} tokens actual vs {first.estimated_tokens} "
                       f"estimated (estimate is {first.estimated_tokens / first.prompt_tokens:.2f}x actual)")
        if sug:
            t, lo, hi = sug
            caught = sum(s >= t for s in by["yes"])
            fp = sum(s >= t for s in by["no"])
            maybes = sum(s >= t for s in by["maybe"])
            out.append(f"  suggested threshold: {t}  (best range {lo}-{hi}; at {t}: {caught}/{len(by['yes'])} yes alerted, "
                       f"{fp}/{len(by['no'])} no alerted, {maybes}/{len(by['maybe'])} maybe alerted)")
            cur_c = sum(s >= current_threshold for s in by["yes"])
            cur_fp = sum(s >= current_threshold for s in by["no"])
            out.append(f"  current threshold {current_threshold}: {cur_c}/{len(by['yes'])} yes, {cur_fp}/{len(by['no'])} no alerted")
        if dropped:  # what would actually be sent: rules first, then the threshold
            alerted = {lab: sum(1 for i in items if i.label == lab and not dropped.get(i.id)
                                and (o := r.outcomes.get(i.id)) and o.score is not None
                                and o.score >= current_threshold) for lab in LABELS}
            n = {lab: sum(i.label == lab for i in items) for lab in LABELS}
            out.append(f"  rules + threshold {current_threshold}: {alerted['yes']}/{n['yes']} yes, "
                       f"{alerted['maybe']}/{n['maybe']} maybe, {alerted['no']}/{n['no']} no alerted")
    return "\n".join(out)


def save_results(cal_dir: Path, items: list[LabelledJob], runs: list[ModelRun], cv_source: str,
                 dropped: dict[str, str | None] | None = None) -> Path:
    dropped = dropped or {}
    data = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cv": cv_source,
        "models": {r.name: {"unavailable": r.unavailable, "results": {
            i.id: {"label": i.label, "title": i.job.title, "rules": dropped.get(i.id) or "pass", **({
                "score": o.score, "seconds": round(o.seconds, 2), "error": o.error, "prompt_tokens": o.prompt_tokens,
                "reason": o.result.reason if o.result else None,
                "scores": o.result.scores if o.result else None,
                "missing": o.result.missing_requirements if o.result else None} if (o := r.outcomes.get(i.id)) else {})}
            for i in items}} for r in runs},
    }
    path = cal_dir / "results.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


async def run_calibration(cfg: Config, models: list[str], cv_path: Path | None) -> str:
    cal = calibration_dir(cfg)
    items = load_labelled(cal)
    if not items:
        raise SystemExit(f"No labelled jobs found in {cal / 'labels.yaml'}")
    cv_file = cv_path or cfg.cv_path
    cv_text = load_cv_text(cv_file, cfg.data_dir)
    if not cv_text:
        raise SystemExit(f"No CV text at {cv_file}; pass --cv <file.pdf|file.txt>")
    runs = [await score_with_model(cfg, cv_text, items, m) for m in (models or [None])]
    dropped = rule_verdicts(cfg, items)
    text = report(items, runs, cfg.notify.min_score, dropped)
    path = save_results(cal, items, runs, str(cv_file), dropped)
    counts = {lab: sum(i.label == lab for i in items) for lab in LABELS}
    return (f"Calibration: {len(items)} jobs ({counts['yes']} yes, {counts['maybe']} maybe, {counts['no']} no), "
            f"CV: {cv_file}\n{text}\n\nPer-job reasons saved to {path}")
