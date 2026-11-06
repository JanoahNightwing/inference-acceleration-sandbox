"""Runs the speculative_decoding.c binary in JSON-events mode and streams its events.

The binary is started with `-j 1`, which makes it print one JSON object per line:
  start   -> model configs + run settings
  prompt  -> decoded prompt pieces and prefill time
  draft   -> one drafted token (text, draft prob q, time since step start)
  verify  -> per-token verdicts (accepted / rejected / discarded with p, q),
             the token the target model contributed, and step timings
  done    -> totals
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EngineSettings:
    engine_dir: Path = field(default_factory=lambda: Path(os.environ.get("SD_ENGINE_DIR", ROOT / "engine")))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("SD_MODELS_DIR", ROOT / "models")))
    max_concurrent_runs: int = int(os.environ.get("SD_MAX_RUNS", "2"))

    @property
    def binary(self) -> Path:
        return self.engine_dir / "speculative_decoding"

    @property
    def tokenizer(self) -> Path:
        return self.engine_dir / "tokenizer.bin"

    def list_models(self) -> list[dict]:
        if not self.models_dir.is_dir():
            return []
        out = []
        for p in sorted(self.models_dir.glob("*.bin")):
            out.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
        return out

    def status(self) -> dict:
        models = self.list_models()
        reason = None
        if not self.binary.is_file():
            reason = f"engine binary not built: run `make` in {self.engine_dir}"
        elif not self.tokenizer.is_file():
            reason = f"tokenizer.bin missing in {self.engine_dir}"
        elif len(models) < 2:
            reason = f"need a base and a draft checkpoint in {self.models_dir}"
        by_size = [m["name"] for m in sorted(models, key=lambda m: m["size_mb"])]
        return {
            "engine_available": reason is None,
            "reason": reason,
            "models": models,
            # known tinyllamas pairs first, otherwise the largest checkpoint verifies the smallest
            "default_base": _pick(by_size, ["stories110M.bin", "stories42M.bin"], fallback=-1),
            "default_draft": _pick(by_size, ["stories15M.bin", "stories260K.bin"], fallback=0),
        }


def _pick(names: list[str], preferred: list[str], fallback: int) -> str | None:
    for p in preferred:
        if p in names:
            return p
    return names[fallback] if names else None


@dataclass
class RunParams:
    prompt: str = "Once upon a time"
    steps: int = 200
    n_draft: int = 4
    temperature: float = 1.0
    topp: float = 0.9
    seed: int = 0
    base_model: str | None = None
    draft_model: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        def num(key, default, lo, hi, cast):
            try:
                v = cast(d.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        return cls(
            prompt=str(d.get("prompt", cls.prompt))[:500],
            steps=num("steps", cls.steps, 1, 1024, int),
            n_draft=num("n_draft", cls.n_draft, 0, 12, int),
            temperature=num("temperature", cls.temperature, 0.0, 2.0, float),
            topp=num("topp", cls.topp, 0.0, 1.0, float),
            seed=num("seed", cls.seed, 0, 2**31 - 1, int),
            base_model=d.get("base_model") or None,
            draft_model=d.get("draft_model") or None,
        )


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, settings: EngineSettings | None = None):
        self.settings = settings or EngineSettings()
        self._slots = asyncio.Semaphore(self.settings.max_concurrent_runs)

    def _resolve_model(self, name: str | None, default: str | None) -> Path:
        name = name or default
        allowed = {m["name"] for m in self.settings.list_models()}
        if not name or name not in allowed:  # only files listed in models_dir, never arbitrary paths
            raise EngineError(f"unknown model: {name!r}")
        return self.settings.models_dir / name

    def build_command(self, params: RunParams) -> list[str]:
        status = self.settings.status()
        if not status["engine_available"]:
            raise EngineError(status["reason"])
        base = self._resolve_model(params.base_model, status["default_base"])
        draft = self._resolve_model(params.draft_model, status["default_draft"])
        seed = params.seed or int.from_bytes(os.urandom(3), "little") + 1
        return [
            str(self.settings.binary),
            "-m", str(base), "-d", str(draft),
            "-z", str(self.settings.tokenizer),
            "-n", str(params.steps), "-k", str(params.n_draft),
            "-t", f"{params.temperature:g}", "-p", f"{params.topp:g}",
            "-s", str(seed), "-i", params.prompt, "-j", "1",
        ]

    async def stream(self, params: RunParams) -> AsyncIterator[dict]:
        cmd = self.build_command(params)
        async with self._slots:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(self.settings.engine_dir),
            )
            try:
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # ignore stray non-JSON output
                    if event.get("type") == "start":
                        event["source"] = "engine"
                        event["base_name"] = Path(cmd[cmd.index("-m") + 1]).name
                        event["draft_name"] = Path(cmd[cmd.index("-d") + 1]).name
                    yield event
                code = await proc.wait()
                if code != 0:
                    err = (await proc.stderr.read()).decode(errors="replace").strip() if proc.stderr else ""
                    yield {"type": "error", "message": err or f"engine exited with code {code}"}
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
