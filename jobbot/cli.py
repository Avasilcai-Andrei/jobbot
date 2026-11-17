"""Command line: python -m jobbot {run,once,report,recent,dry-run,test-notify,heartbeat,calibrate,...}"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from . import digest as digest_mod, guard as guard_mod, heartbeat, ollama_ctl, status as status_mod
from .app import channels, open_pipeline
from .calibrate import add_job, import_marks, run_calibration
from .config import Config, load_config
from .control import Control, describe
from .db import DB, utcnow
from .http import PoliteClient
from .models import Job
from .notify import Alert
from .pipeline import evaluate_current
from .report import build_marks_report, build_report

log = logging.getLogger("jobbot")


async def cmd_once(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    async with open_pipeline(cfg, console=args.console) as pipeline:
        stats = await pipeline.run_once()
    log.info("Run finished: %s", stats.summary())
    return 0


EXIT_CONFIG = 2    # don't restart: fix the configuration first (the autostart script stops on this)
EXIT_RUNNING = 3   # another `run` already holds the lock
# `pause`, `unload`: Ollama is not reachable (so no model is loaded) / the model is still loaded after the attempt.
EXIT_OLLAMA_DOWN = 4
EXIT_STILL_LOADED = 5
# `status`: 0 running, 10 scoring deferred, 11 paused (scriptable: `jobbot status` before using the GPU).
EXIT_STATUS = {"running": 0, "deferred": 10, "paused": 11}


def acquire_lock(path: Path):
    """Hold an exclusive lock on `path` for the life of the process; None if another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


async def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not args.console:
        async with PoliteClient(cfg.http.user_agent) as http:
            if not channels(cfg, http):
                # Running without a channel would mark new jobs "notified" while nobody sees them.
                log.error("No notification channel: set DISCORD_WEBHOOK_URL (and/or SMTP_USER + SMTP_APP_PASSWORD) "
                          "in %s, or use --console", cfg.base_dir / ".env")
                return EXIT_CONFIG
    lock = acquire_lock(cfg.data_dir / "jobbot.lock")
    if lock is None:
        log.error("Another `jobbot run` is already running (lock %s); exiting", cfg.data_dir / "jobbot.lock")
        return EXIT_RUNNING
    interval = cfg.schedule.interval_minutes * 60
    async with open_pipeline(cfg, console=args.console) as pipeline:
        log.info("jobbot running every %.0f min with %d feeds", interval / 60, len(pipeline.sources))
        control = pipeline.control
        beacon = asyncio.create_task(_beacon(control, cfg.control.poll_seconds))
        sampler = asyncio.create_task(pipeline.guard.run())  # GPU samples every control.gpu_sample_seconds
        try:
            while True:
                started = time.monotonic()
                started_at = utcnow()
                control.in_run = True
                try:
                    stats = await pipeline.run_once()
                    if stats.paused:
                        log.info("Run skipped: paused")
                    else:
                        log.info("Run finished in %.1fs: %s", time.monotonic() - started, stats.summary())
                except Exception:
                    log.exception("Run failed")
                    pipeline.db.record_run(started_at, utcnow(), {"crashed": 1})
                finally:
                    control.in_run = False
                if not control.read().paused:  # nothing is sent while paused; the heartbeat catches up afterwards
                    await heartbeat.maybe_send(cfg, pipeline.db, pipeline.notifier, utcnow())
                    await digest_mod.maybe_send(pipeline, utcnow())
                wait = max(5.0, interval - (time.monotonic() - started))
                control.next_run_at = utcnow() + timedelta(seconds=wait)
                await _idle(pipeline, wait, cfg.control.poll_seconds)
        finally:
            beacon.cancel()
            sampler.cancel()


async def _beacon(control: Control, poll: float) -> None:
    """Tell `status` and the panel that the bot is alive, also while a run or a model load blocks the loop."""
    while True:
        try:
            control.write_bot_status()
        except Exception:
            log.exception("Could not write the bot status")
        await asyncio.sleep(poll)


async def _idle(pipeline, seconds: float, poll: float) -> None:
    """Wait for the next scan, watching the control state: unload the model when a pause starts, return early
    when a pause ends (run now), and score the waiting jobs when a deferral ends."""
    control = pipeline.control
    state = control.refresh()
    if state.paused:
        await pipeline.unload_model()  # a score that was in flight when the pause began may have loaded it again
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while (left := deadline - loop.time()) > 0:
        await asyncio.sleep(min(poll, left))
        cur = control.refresh()
        if cur.paused and not state.paused:
            log.info("Pause started: unloading the model")
            await pipeline.unload_model()
        elif state.paused and not cur.paused:
            log.info("Pause ended: running now to catch up")
            return
        elif state.deferred and not cur.deferred and not cur.paused:
            await pipeline.drain_pending()
        elif pipeline.guard.cleared and not cur.paused and not cur.deferred:
            pipeline.guard.cleared = False  # the GPU just went quiet: score what waited without waiting for the scan
            if pipeline.db.count_status("pending"):
                await pipeline.drain_pending()
        state = cur


async def _unload_now(cfg: Config, quiet: bool = False) -> int:
    """Unload the configured Ollama model through the server, so it works while the bot is stopped too."""
    if cfg.llm.backend != "ollama":
        return 0
    o = cfg.llm.ollama
    say = (lambda _: None) if quiet else print
    try:
        async with httpx.AsyncClient(base_url=o.url.rstrip("/"), timeout=5.0) as client:
            if await ollama_ctl.loaded_model(client, o.model) is None:
                say(f"Model {o.model} is not loaded.")
                return 0
            if await ollama_ctl.unload(client, o.model):
                say(f"Unloaded {o.model}.")
                return 0
    except httpx.HTTPError as e:
        say(f"Ollama is not reachable at {o.url} ({type(e).__name__}); no model to unload.")
        return EXIT_OLLAMA_DOWN
    say(f"{o.model} is still loaded after the unload request.")
    return EXIT_STILL_LOADED


def _local(t: datetime) -> str:
    return t.astimezone().strftime("%H:%M")


def _control(args: argparse.Namespace) -> tuple[Config, DB, Control]:
    cfg = load_config(args.config)
    db = DB(cfg.db_path)
    return cfg, db, Control(db, cfg)


async def cmd_pause(args: argparse.Namespace) -> int:
    cfg, db, control = _control(args)
    try:
        state = control.pause(args.hours)
        if state.paused_until:
            print(f"Paused until {_local(state.paused_until)} (auto-resume). `python -m jobbot resume` ends it sooner.")
        else:
            print("Paused until you run `python -m jobbot resume`.")
    finally:
        db.close()
    rc = await _unload_now(cfg)
    return 0 if rc in (0, EXIT_OLLAMA_DOWN) else rc  # Ollama not running = nothing holds the GPU


def cmd_resume(args: argparse.Namespace) -> int:
    _, db, control = _control(args)
    try:
        before, _ = control.resume()
        print("Resumed." if before.mode != "running" else "Already running.")
    finally:
        db.close()
    return 0


def cmd_defer(args: argparse.Namespace) -> int:
    _, db, control = _control(args)
    try:
        if args.minutes is not None:
            state = control.defer("minutes", args.minutes)
        else:
            state = control.defer("resume" if args.until_resume else "next_scan")
        print(f"{describe(state, utcnow()).capitalize()}. Fetching and filtering continue; matching jobs wait.")
    finally:
        db.close()
    return 0


def cmd_busy(args: argparse.Namespace) -> int:
    cfg, db, _ = _control(args)
    try:
        now, c = utcnow(), cfg.control
        if args.clear:
            print("Busy flag cleared." if guard_mod.clear_flag(db) else "No busy flag was set.")
        elif args.set:
            guard_mod.set_flag(db, args.set, args.hours, now)
            until = f"for {args.hours:g} h" if args.hours else f"at most {c.busy_max_hours:g} h (control.busy_max_hours)"
            print(f"Busy flag set ({args.set}); scoring waits, {until}. `python -m jobbot busy --clear` ends it.")
        flag, stale = guard_mod.read_flag(db, now, c.busy_max_hours)
        lock, lock_stale = guard_mod.read_lock(cfg.resolve(Path(c.busy_lock_path)), now, c.busy_max_hours)
        print(f"Flag: {flag or 'none'}" + (f" (ignored: {stale})" if stale else ""))
        print(f"Lock file {cfg.resolve(Path(c.busy_lock_path))}: {lock or 'none'}" + (f" (ignored: {lock_stale})" if lock_stale else ""))
    finally:
        db.close()
    return 0


async def cmd_unload(args: argparse.Namespace) -> int:
    return await _unload_now(load_config(args.config))


async def cmd_status(args: argparse.Namespace) -> int:
    cfg, db, control = _control(args)
    try:
        now = utcnow()
        info = await status_mod.gather(cfg, control, now)
    finally:
        db.close()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(status_mod.format_text(info, now))
    return EXIT_STATUS[info["mode"]]


def cmd_panel(args: argparse.Namespace) -> int:
    from .panel import run_panel
    cfg = load_config(args.config)
    root = logging.getLogger()
    root.handlers[:] = [logging.NullHandler()]  # log lines on stderr would draw over the screen
    return run_panel(cfg)


async def cmd_heartbeat(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    async with open_pipeline(cfg, console=args.console, use_llm=False) as pipeline:
        text = heartbeat.build_message(pipeline.db, utcnow())
        await pipeline.notifier.send_text(text)
    print("Heartbeat sent." if not args.console else "")
    return 0


async def cmd_digest(args: argparse.Namespace) -> int:
    """Preview the email digest for the email jobs listed right now. Reads the mailbox read-only, writes nothing."""
    cfg = load_config(args.config)
    async with open_pipeline(cfg, console=True, use_llm=args.llm) as pipeline:
        pipeline.sources = [s for s in pipeline.sources if s.kind == "email"]
        if not pipeline.sources:
            print("No email source is configured (sources.email, JOBMAIL_* in .env).")
            return 0
        digest, plan, rejected = await digest_mod.preview(pipeline, use_llm=args.llm)
        unique_rejected = {j.key: v for j, v in rejected}   # the same posting recurs in several digests
        reasons: dict[str, int] = {}
        for v in unique_rejected.values():
            key = v.reason.split("'")[0].strip() or v.reason
            reasons[key] = reasons.get(key, 0) + 1
        total = len(unique_rejected) + len({j.key for j, _ in plan.keep}) + len({j.key for j, _, _ in plan.dropped})
        print(f"{total} distinct email postings; the rule filter dropped {len(unique_rejected)}:")
        for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:4}  {k}")
        print(f"{len(plan.dropped)} dropped as duplicates:")
        for job, reason, _ in plan.dropped:
            print(f"        {job.title[:50]} @ {job.company[:25]}: {reason}")
        if digest is None:
            print("Nothing would be sent.")
            return 0
        print(f"\nWould send {len(digest.items)} job(s), as the messages below (the notifier would post each one):")
        await pipeline.notifier.send_digest(digest)
    return 0


async def cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    async with open_pipeline(cfg, console=True, use_llm=args.llm) as pipeline:
        evals = await evaluate_current(pipeline, use_llm=args.llm)
    shown = evals if args.all else [e for e in evals if e.verdict.ok or e.verdict.role]
    for e in sorted(shown, key=lambda e: (not e.verdict.ok, e.job.feed, e.job.title)):
        mark = "PASS" if e.verdict.ok else "drop"
        prio = f" [{e.verdict.priority}]" if e.verdict.ok else ""
        print(f"{mark}{prio:7} {e.job.feed:28} {e.job.title[:60]:60} | {e.job.location[:40]}")
        if not e.verdict.ok:
            print(f"        -> {e.verdict.reason}")
        if e.match:
            twice = f" (scored twice: {e.match.scores[0]} and {e.match.scores[1]})" if e.match.scores else ""
            print(f"        -> score {e.match.score}{twice}: {e.match.reason}")
        if e.error:
            print(f"        -> LLM error: {e.error}")
    passed = sum(e.verdict.ok for e in evals)
    print(f"\n{len(evals)} jobs listed, {passed} pass the rule filter"
          + ("" if args.all else " (use --all to list every dropped job)"))
    if args.llm:
        scored = [e for e in evals if e.match]
        print(f"{len(scored)} scored, {sum(bool(e.match.scores) for e in scored)} of them twice "
              f"(first score within ±{cfg.llm.rescore_margin} of the threshold)")
    return 0


async def cmd_test_notify(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    async with open_pipeline(cfg, console=args.console, use_llm=False) as pipeline:
        now = utcnow()
        job = Job(source="test", job_id="0", feed="test:jobbot", title="Junior Data Engineer (test alert)",
                  company="jobbot", location="Cluj-Napoca, Romania (hybrid)", url="https://example.com/jobbot-test",
                  posted_at=now.replace(microsecond=0))
        await pipeline.notifier.send_alert(Alert(
            job=job, score=88, reason="This is a test alert from jobbot. If you can read it, notifications work.",
            missing=["Example missing requirement"], location_rule="Cluj-Napoca", low_priority=False,
            first_seen_at=now))
    print("Test alert sent.")
    return 0


async def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(await run_calibration(cfg, args.model or [], Path(args.cv) if args.cv else None))
    return 0


def cmd_calibrate_import_marks(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db = DB(cfg.db_path)
    try:
        rows = import_marks(cfg, db)
    finally:
        db.close()
    for sid, label, title, outcome in rows:
        print(f"{label:3}  {outcome:28} {title[:60]}  ({sid[:50]})")
    print(f"{len(rows)} marked alert(s); labels.yaml: good = yes, bad = no")
    return 0


async def cmd_calibrate_add(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = await add_job(cfg, args.feed, args.job_id, args.label, args.note or "", args.location)
    print(f"Saved {path}; check its label in calibration/labels.yaml")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db = DB(cfg.db_path)
    try:
        print(build_marks_report(db, days=args.days) if args.marks else build_report(db, days=args.days, by=args.by,
                                                                                interval_minutes=cfg.schedule.interval_minutes))
    finally:
        db.close()
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db = DB(cfg.db_path)
    try:
        short = args.id.strip().lstrip("#").lower()
        rows = db.alerts_by_short_id(short)
        if len(rows) != 1:
            print(f"No alert with id {short}." if not rows else
                  f"Id {short} matches {len(rows)} alerts: " + "; ".join(f"{r['feed']} {r['job_id']}" for r in rows))
            print("`python -m jobbot report --marks` lists the alerts and their ids.")
            return 1
        r = rows[0]
        note = " ".join(args.note) or None
        db.set_mark((r["source"], r["job_id"]), args.label, note)
        score = r["score"] if r["score"] is not None else "unscored"
        print(f"Marked {short} {args.label}: {r['title']} @ {r['company']} (score {score}, "
              f"alerted {r['notified_at'][:16].replace('T', ' ')} UTC){f' - {note}' if note else ''}")
    finally:
        db.close()
    return 0


def cmd_recent(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db = DB(cfg.db_path)
    try:
        for r in db.recent(args.limit, args.status):
            seen = utcnow() - datetime.fromisoformat(r["first_seen_at"])
            score = f"{r['score']:>3}" if r["score"] is not None else "  -"
            print(f"{r['status']:9} {score} {r['feed']:26} {r['title'][:55]:55} "
                  f"{int(seen.total_seconds() // 60):>6}m ago  {r['reason'] or ''}")
    finally:
        db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="python -m jobbot", description="Job-alert bot")
    parser.add_argument("--config", default=os.environ.get("JOBBOT_CONFIG", "config.yaml"))
    parser.add_argument("--log-level", default=os.environ.get("JOBBOT_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", default=os.environ.get("JOBBOT_LOG_FILE"),
                        help="log to this file instead of stderr (rotates at 5 MB, keeps 5 old files)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_ in (("run", "run forever on the configured interval"), ("once", "run the pipeline once")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--console", action="store_true", help="print alerts instead of sending them")
    p = sub.add_parser("report", help="detection-latency percentiles per source")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--by", choices=["source", "feed"], default="source")
    p.add_argument("--marks", action="store_true", help="list alerts with your good/bad marks instead")
    p = sub.add_parser("mark", help="mark an alert good or bad (the id is in the alert)")
    p.add_argument("id", help="the alert's short id, e.g. 3fa9c2")
    p.add_argument("label", choices=["good", "bad"])
    p.add_argument("note", nargs="*", help="optional note, e.g. why it's bad")
    p = sub.add_parser("recent", help="list recently seen jobs and what happened to them")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--status", choices=["seeded", "new", "filtered", "duplicate", "scored", "notified", "error"])
    p = sub.add_parser("dry-run", help="evaluate everything currently listed; sends nothing, stores nothing")
    p.add_argument("--llm", action="store_true", help="also score rule survivors with the configured scorer")
    p.add_argument("--all", action="store_true", help="also list jobs dropped for role/title reasons")
    p = sub.add_parser("test-notify", help="send a sample alert to check the webhook")
    p.add_argument("--console", action="store_true")
    p = sub.add_parser("heartbeat", help="send the daily heartbeat message now")
    p.add_argument("--console", action="store_true")
    p = sub.add_parser("pause", help="stop everything (no fetching, scoring or sending) and unload the model")
    p.add_argument("--hours", type=float, default=None,
                   help="auto-resume after this long (default control.pause_auto_resume_hours; 0 = until resume)")
    sub.add_parser("resume", help="end a pause and any scoring deferral")
    p = sub.add_parser("defer", help="keep fetching and filtering but don't score (jobs wait; the model isn't loaded)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--minutes", type=float, help="for N minutes")
    g.add_argument("--next-scan", action="store_true", help="until the next scan starts (the default)")
    g.add_argument("--until-resume", action="store_true", help="until `jobbot resume`")
    sub.add_parser("panel", help="open the control panel (a terminal UI; closing it doesn't affect the bot)")
    p = sub.add_parser("busy", help="tell jobbot the GPU is in use (scoring waits): --set REASON [--hours N] | --clear")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--set", metavar="REASON")
    g.add_argument("--clear", action="store_true")
    p.add_argument("--hours", type=float, default=None, help="with --set: the flag ends by itself after this long")
    sub.add_parser("unload", help="unload the Ollama model now")
    p = sub.add_parser("status", help="pause/defer state, bot, model and GPU (exit code: 0 running, 10 deferred, 11 paused)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("calibrate", help="score the labelled jobs in calibration/ and suggest a threshold")
    p.add_argument("--model", action="append", help="model to test (repeatable); default: the configured one")
    p.add_argument("--cv", help="CV file (.pdf or .txt); default: profile.cv_path")
    sub.add_parser("calibrate-import-marks", help="copy your good/bad alert marks into the calibration set")
    p = sub.add_parser("digest", help="preview the daily email digest (reads the mailbox read-only, sends nothing)")
    p.add_argument("--llm", action="store_true", help="ask the local model for the title-fit hints")
    p = sub.add_parser("calibrate-add", help="save a currently listed job into the calibration set")
    p.add_argument("--feed", required=True, help="e.g. smartrecruiters:Endava, greenhouse:databricks")
    p.add_argument("--job-id", required=True)
    p.add_argument("--label", choices=["yes", "maybe", "no"], default="maybe")
    p.add_argument("--note")
    p.add_argument("--location", help="judge the role as if it were here (original location is kept)")

    args = parser.parse_args(argv)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            args.log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        logging.basicConfig(level=args.log_level.upper(), format=fmt, handlers=[handler])
    else:
        logging.basicConfig(level=args.log_level.upper(), format=fmt)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.Formatter.converter = time.gmtime  # log in UTC, like the database

    handlers = {"run": cmd_run, "once": cmd_once, "dry-run": cmd_dry_run, "test-notify": cmd_test_notify,
                "calibrate": cmd_calibrate, "calibrate-add": cmd_calibrate_add, "digest": cmd_digest, "heartbeat": cmd_heartbeat,
                "pause": cmd_pause, "unload": cmd_unload, "status": cmd_status}

    def stop(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGTERM", "SIGBREAK"):  # SIGBREAK: Ctrl+Break / console close on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    try:
        if args.command in handlers:
            return asyncio.run(handlers[args.command](args))
        return {"report": cmd_report, "recent": cmd_recent, "mark": cmd_mark,
                "calibrate-import-marks": cmd_calibrate_import_marks,
                "resume": cmd_resume, "defer": cmd_defer, "panel": cmd_panel, "busy": cmd_busy}[args.command](args)
    except KeyboardInterrupt:
        # The DB commits after every write, so stopping here loses nothing.
        log.info("Stopped")
        return 130
    except (FileNotFoundError, ValidationError, yaml.YAMLError) as e:
        if isinstance(e, FileNotFoundError) and not str(e.filename or "").endswith((".yaml", ".yml")):
            raise
        log.error("Configuration problem: %s", e)
        return EXIT_CONFIG

