"""Ask the Ollama server what is loaded and unload a model. Works without the bot (the CLI uses it directly)."""
from __future__ import annotations

import asyncio
import os
import subprocess
import logging
from pathlib import Path
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)


def _names(entry: dict[str, Any]) -> set[str]:
    return {entry.get("name"), entry.get("model")} - {None}


async def loaded_model(client: httpx.AsyncClient, model: str) -> dict[str, Any] | None:
    """The /api/ps entry for `model` (name, size, size_vram, expires_at), None if it isn't loaded.
    Raises httpx.HTTPError when the server can't be reached."""
    resp = await client.get("/api/ps", timeout=5.0)
    resp.raise_for_status()
    for m in resp.json().get("models", []):
        if model in _names(m) or f"{model}:latest" in _names(m):
            return m
    return None


async def unload(client: httpx.AsyncClient, model: str, wait: float = 10.0) -> bool:
    """Unload `model` now (keep_alive 0) and wait until /api/ps agrees. True if it is no longer loaded.
    Raises httpx.HTTPError when the server can't be reached."""
    if await loaded_model(client, model) is None:  # don't send keep_alive 0 for a model that isn't loaded
        return True
    resp = await client.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=15.0)
    if resp.status_code >= 400 and resp.status_code != 404:  # 404: model not pulled, so it isn't loaded either
        log.warning("Ollama answered HTTP %d to the unload request", resp.status_code)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait
    while True:
        if await loaded_model(client, model) is None:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.5)


# --- starting the server ---------------------------------------------------------------------------------

def _install_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama"


def _process_running(name: str) -> bool:
    import psutil
    return any((p.info["name"] or "").lower() == name.lower() for p in psutil.process_iter(["name"]))


def launch_server() -> str | None:
    """Start Ollama the way scripts/jobbot-autostart.ps1 does: the tray app if it is installed (it manages the
    server), else `ollama serve`. Returns what was started, or None when nothing could be (not Windows, not
    installed, or the tray app is already running and only needs time). The tray app goes through explorer.exe so
    it is not a child of the bot (stopping the bot's process tree must not take Ollama down with it)."""
    if os.name != "nt":
        return None
    tray, cli = _install_dir() / "ollama app.exe", _install_dir() / "ollama.exe"
    if tray.exists():
        if _process_running("ollama app.exe"):
            return None
        subprocess.Popen(["explorer.exe", str(tray)], close_fds=True)
        return "the Ollama tray app"
    if cli.exists():
        if _process_running("ollama.exe"):
            return None
        subprocess.Popen([str(cli), "serve"], close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=0x00000008 | 0x08000000)  # DETACHED | NO_WINDOW
        return "ollama serve"
    return None


async def start_and_wait(client: httpx.AsyncClient, wait: float,
                         launcher: Callable[[], str | None] = launch_server, poll: float = 2.0) -> tuple[bool, str]:
    """Try to bring a server that doesn't answer up, and wait up to `wait` seconds for it. (answers, note)."""
    try:
        started = await asyncio.to_thread(launcher)
    except Exception as e:  # noqa: BLE001 - starting is best effort; the caller falls back to holding the jobs
        return False, f"could not start Ollama ({type(e).__name__}: {e})"
    note = f"started {started}" if started else "no new process started (not installed, or already starting)"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait
    while True:
        try:
            resp = await client.get("/api/version", timeout=3.0)
            if resp.status_code < 500:
                return True, note
        except httpx.HTTPError:
            pass
        if loop.time() >= deadline:
            return False, f"{note}; still no answer after {wait:.0f} s"
        await asyncio.sleep(poll)
