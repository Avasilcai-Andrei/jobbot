"""Stand-ins for nvidia-smi, Ollama's /api/ps and the process list, driven by one mutable `Rig`."""
from __future__ import annotations

from jobbot.gpu import GpuReading
from jobbot.guard import GpuGuard

OWN = "qwen3.5:9b"


class Rig:
    def __init__(self):
        self.util = 2
        self.free_mb = 10_500
        self.smi_ok = True
        self.models: list[str] = []
        self.ollama_ok = True
        self.procs: list[tuple[int, str, str]] = [(10, "explorer.exe", "C:\\Windows\\explorer.exe")]
        self.alive: set[int] = set()

    async def read_gpu(self):
        return GpuReading(self.util, 12_000 - self.free_mb, self.free_mb, ["explorer.exe"]) if self.smi_ok else None

    async def loaded_models(self):
        return list(self.models) if self.ollama_ok else None


def make_guard(cfg, db, clock, rig: Rig, own: str | None = OWN) -> GpuGuard:
    return GpuGuard(cfg, db, own_model=own, clock=clock, read_gpu=rig.read_gpu, loaded_models=rig.loaded_models,
                    processes=lambda: list(rig.procs), pid_is_alive=lambda pid: pid in rig.alive)
