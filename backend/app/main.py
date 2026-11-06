"""Inference Acceleration Sandbox -- FastAPI wrapper around speculative_decoding.c.

  GET  /api/status      engine availability + checkpoints found in models/
  WS   /ws/generate     send {"type":"run", ...params}, receive a stream of events;
                        send {"type":"stop"} to cancel the current run
Serves the built React app (frontend/dist) at / when it exists.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .engine import Engine, EngineError, RunParams
from .simulator import simulate

app = FastAPI(title="Inference Acceleration Sandbox")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
engine = Engine()


@app.get("/api/status")
def status() -> dict:
    return engine.settings.status()


def event_source(msg: dict):
    """Pick the real engine or the simulator for a run request."""
    params = RunParams.from_dict(msg)
    want = msg.get("source", "auto")
    if want == "simulated":
        return simulate(params)
    if want == "engine" or engine.settings.status()["engine_available"]:
        return engine.stream(params)
    return simulate(params)


@app.websocket("/ws/generate")
async def generate(ws: WebSocket) -> None:
    await ws.accept()
    run_task: asyncio.Task | None = None

    async def pump(msg: dict) -> None:
        try:
            async for event in event_source(msg):
                await ws.send_json(event)
        except EngineError as e:
            await ws.send_json({"type": "error", "message": str(e)})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface unexpected failures to the UI instead of dropping the socket
            await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def cancel() -> None:
        nonlocal run_task
        if run_task and not run_task.done():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
        run_task = None

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "run":
                await cancel()
                run_task = asyncio.create_task(pump(msg))
            elif kind == "stop":
                await cancel()
                await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        pass
    finally:
        await cancel()


DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
