"""Control panel (`python -m jobbot panel`): a Textual app that shows the bot's state and sends it commands.

It only reads the database, Ollama and nvidia-smi, and writes the same control state as `jobbot pause|defer|resume`.
Closing it never affects the bot, which keeps running under Task Scheduler.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable

from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Static

from .config import Config
from .control import Control, describe
from .db import DB, utcnow
from .status import _ago, _in, gather, gpu_line, pending_lines

MODE_STYLE = {"running": "green", "deferred": "yellow", "paused": "red"}


def _t(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:  # Ollama sends up to 9 fractional digits
        head, _, rest = iso.partition(".")
        tz = rest[next((i for i, c in enumerate(rest) if c in "+-Z"), len(rest)):]
        return datetime.fromisoformat(head + tz.replace("Z", "+00:00"))


def _hm(t: datetime) -> str:
    return t.astimezone().strftime("%H:%M")


def _next_load(info: dict[str, Any], now: datetime) -> str:
    nxt = _t(info["next_run_at"])
    scan = f"next scan {_hm(nxt)} (in {_in(nxt, now)})" if nxt else "next scan unknown"
    if not info["bot_alive"]:
        return "none: the bot is not running"
    if info["mode"] == "paused":
        return "after you resume"
    kind = info["defer_kind"]
    if kind == "resume":
        return "after you resume"
    if kind == "minutes":
        end = _t(info["defer_until"])
        return f"when the deferral ends at {_hm(end)} (in {_in(end, now)})"
    return ("a scan is running now" if info["in_run"] else scan) + f" · {info['pending']} job(s) pending"


def render(info: dict[str, Any], now: datetime) -> dict[str, str]:
    """Markup for each panel section (pure, so it is testable without a terminal)."""
    mode = info["mode"]
    state = f"[b {MODE_STYLE[mode]}]{mode.upper()}[/]  {escape(info['description'])}"
    seen = _t(info["bot_last_seen"])
    if info["bot_alive"]:
        nxt = _t(info["next_run_at"])
        doing = "scanning now" if info["in_run"] else f"next scan {_hm(nxt)}, in {_in(nxt, now)}" if nxt else "idle"
        bot = f"[green]alive[/] (pid {info['bot_pid']}) · {doing}"
    else:
        bot = "[b red]NOT RUNNING[/]" + (f" · last seen {_ago(seen, now)}" if seen else " · never seen")
    lr = info.get("last_run")
    if lr:
        done = _t(info["last_run_finished"])
        bot += f"\nlast run {_ago(done, now)}"

    m = info["model"]
    if m is None:
        model = "no local model (the scoring backend is not ollama)"
    elif info["ollama_reachable"] is False:
        model = f"{m['name']}: [red]Ollama not reachable[/]"
    elif m["loaded"]:
        exp = _t(m.get("expires_at"))
        vram = f"{m['size_vram'] / 1024 ** 3:.1f} GB VRAM" if m.get("size_vram") else "VRAM unknown"
        model = f"{m['name']}: [b]loaded[/], {vram}" + (f", until {_hm(exp)} (in {_in(exp, now)})" if exp else "")
    else:
        model = f"{m['name']}: not loaded"
    free = info["gpu_free_mb"]
    model += "\nGPU free: " + (f"{free / 1024:.1f} GB" if free is not None else "nvidia-smi unavailable")
    gl = gpu_line(info, now)
    if gl:
        model += "\n" + (f"[b red]{escape(gl)}[/]" if info["gpu"]["busy"] else f"[green]{escape(gl)}[/]")
    model += f"\nNext possible model load: {_next_load(info, now)}"

    scoring = "\n".join(escape(line) for line in pending_lines(info, now))

    if lr is None:
        last = "no run recorded yet"
    elif lr.get("crashed"):
        last = "[red]the last run crashed[/]"
    else:
        last = (f"fetched {lr.get('fetched', 0)} · new {lr.get('new', 0)} · passed rules {lr.get('passed_rules', 0)} · "
                f"scored {lr.get('scored', 0)} · alerted {lr.get('notified', 0)}")
        if lr.get("held"):
            last += f" · held {lr['held']}"
    failing = info["failing_feeds"]
    last += "\n" + ("[green]all feeds OK[/]" if not failing else "[red]failing:[/] " + escape(", ".join(
        f"{f['feed']} ({f['failures']}×)" for f in failing[:5])) + (f" +{len(failing) - 5} more" if len(failing) > 5 else ""))

    rows = []
    for a in info["alerts"]:
        when = _t(a["notified_at"]).astimezone().strftime("%d %b %H:%M")
        score = f"{a['score']:>3}" if a["score"] is not None else "  ?"
        mark = {"good": " [green]good[/]", "bad": " [red]bad[/]"}.get(a["mark"], "")
        rows.append(f"[b]{a['id']}[/] {score} {escape(a['title'][:42])} @ {escape(a['company'][:18])} · {when}{mark}")
    alerts = "\n".join(rows) or "no alerts yet"
    return {"state": state, "bot": bot, "model": model, "scoring": scoring, "last": last, "alerts": alerts}


class MinutesScreen(ModalScreen[float | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Grid(id="dialog"):
            yield Label("Defer scoring for how many minutes?")
            yield Input(placeholder="e.g. 90", type="number", id="minutes")

    @on(Input.Submitted)
    def _submit(self, event: Input.Submitted) -> None:
        try:
            minutes = float(event.value)
        except ValueError:
            minutes = 0
        self.dismiss(minutes if minutes > 0 else None)


class PanelApp(App):
    TITLE = "jobbot control panel"
    CSS = """
    Screen { layout: vertical; }
    .box { border: round $primary; padding: 0 1; margin: 0 1; height: auto; }
    #alerts { height: auto; }
    MinutesScreen { align: center middle; }
    #dialog { grid-size: 1; grid-rows: auto auto; width: 44; height: auto; padding: 1 2;
              border: thick $primary; background: $surface; }
    """
    BINDINGS = [
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("d", "defer_next", "Defer to next scan"),
        Binding("m", "defer_minutes", "Defer N min"),
        Binding("r", "resume", "Resume"),
        Binding("u", "unload", "Unload model"),
        Binding("q", "quit", "Quit panel"),
    ]

    def __init__(self, cfg: Config, db: DB | None = None, unload: Callable[[Config], Awaitable[int]] | None = None,
                 clock: Callable[[], datetime] = utcnow):
        super().__init__()
        self.cfg = cfg
        self.db = db or DB(cfg.db_path)
        self.control = Control(self.db, cfg, clock)
        self.clock = clock
        if unload is None:
            from .cli import _unload_now
            unload = lambda c: _unload_now(c, quiet=True)  # noqa: E731
        self._unload = unload
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        for key, title in (("state", "State"), ("bot", "Bot"), ("model", "Model and GPU"),
                           ("scoring", "Scoring queue"), ("last", "Last run"), ("alerts", "Last 5 alerts (mark ids)")):
            box = Static("…", id=key, classes="box")
            box.border_title = title
            yield box
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_info()
        self.set_interval(1.0, self.refresh_info)

    async def refresh_info(self) -> None:
        if self._busy:  # the previous refresh (nvidia-smi, Ollama) is still running
            return
        self._busy = True
        try:
            now = self.clock()
            sections = render(await gather(self.cfg, self.control, now), now)
        except Exception as e:  # keep the panel alive whatever one refresh hits
            self.query_one("#state", Static).update(f"[red]refresh failed:[/] {escape(type(e).__name__)}: {escape(str(e))}")
            return
        finally:
            self._busy = False
        for key, text in sections.items():
            self.query_one(f"#{key}", Static).update(text)

    # --- commands: the same state changes as the CLI -----------------------------

    async def _unload_and_report(self) -> None:
        rc = await self._unload(self.cfg)
        self.notify({0: "Model unloaded (or was not loaded).", 4: "Ollama is not reachable; nothing to unload.",
                     5: "The model is still loaded after the unload request."}.get(rc, f"unload exit code {rc}"),
                    severity="warning" if rc == 5 else "information")

    async def action_toggle_pause(self) -> None:
        if self.control.peek().paused:
            self.control.resume()
            self.notify("Resumed. The bot catches up on its next tick.")
        else:
            state = self.control.pause()
            self.notify("Paused" + (f" until {_hm(state.paused_until)}." if state.paused_until else "."))
            await self._unload_and_report()
        await self.refresh_info()

    async def action_resume(self) -> None:
        before, _ = self.control.resume()
        self.notify("Resumed." if before.mode != "running" else "Already running.")
        await self.refresh_info()

    async def action_defer_next(self) -> None:
        self.control.defer("next_scan")
        self.notify("Scoring deferred until the next scan.")
        await self.refresh_info()

    def action_defer_minutes(self) -> None:
        def done(minutes: float | None) -> None:
            if minutes:
                state = self.control.defer("minutes", minutes)
                self.notify(describe(state, self.clock()).capitalize() + ".")
        self.push_screen(MinutesScreen(), done)

    async def action_unload(self) -> None:
        await self._unload_and_report()
        await self.refresh_info()

    def on_unmount(self) -> None:
        self.db.close()


def run_panel(cfg: Config) -> int:
    PanelApp(cfg).run()
    return 0
