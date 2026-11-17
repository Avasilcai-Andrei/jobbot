"""GPU readings through nvidia-smi: free VRAM, utilization and compute processes."""
from __future__ import annotations

import asyncio
import logging
import ntpath
from dataclasses import dataclass, field
from typing import Sequence

log = logging.getLogger(__name__)

QUERY = ("--query-gpu=memory.free", "--format=csv,noheader,nounits")
QUERY_STATE = ("--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits")
QUERY_APPS = ("--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits")


@dataclass
class GpuReading:
    util_pct: int            # the busiest GPU
    used_mb: int
    free_mb: int             # the GPU with the most free memory
    procs: list[str] = field(default_factory=list)   # compute/graphics process names (informational only)


async def _smi(cmd: Sequence[str], args: Sequence[str], timeout: float) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (OSError, asyncio.TimeoutError) as e:
        log.debug("nvidia-smi unavailable: %s", e)
        return None
    return out.decode("utf-8", "replace") if proc.returncode == 0 else None


def _ints(text: str | None, columns: int) -> list[list[int]]:
    rows = []
    for line in (text or "").splitlines():
        try:
            row = [int(float(x.strip())) for x in line.split(",")]
        except ValueError:
            continue
        if len(row) == columns:
            rows.append(row)
    return rows


async def free_vram_mb(cmd: Sequence[str] = ("nvidia-smi",), timeout: float = 10.0) -> int | None:
    """Free VRAM in MiB (the largest value if there are several GPUs); None if it can't be read."""
    rows = _ints(await _smi(cmd, QUERY, timeout), 1)
    return max(r[0] for r in rows) if rows else None


async def read_gpu(cmd: Sequence[str] = ("nvidia-smi",), timeout: float = 10.0) -> GpuReading | None:
    """Utilization, VRAM and process names in one reading; None if nvidia-smi can't be read."""
    rows = _ints(await _smi(cmd, QUERY_STATE, timeout), 3)
    if not rows:
        return None
    names: list[str] = []
    for line in (await _smi(cmd, QUERY_APPS, timeout) or "").splitlines():
        _, _, name = line.partition(",")
        name = ntpath.basename(name.strip())
        if name and name not in names:
            names.append(name)
    return GpuReading(util_pct=max(r[0] for r in rows), used_mb=sum(r[1] for r in rows),
                      free_mb=max(r[2] for r in rows), procs=names)
