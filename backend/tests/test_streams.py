"""Checks both event sources emit a consistent stream. Run: python -m tests.test_streams (from backend/)."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine import Engine, EngineSettings, RunParams  # noqa: E402
from app.simulator import simulate  # noqa: E402


def check(events, steps):
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[1] == "prompt" and types[-1] == "done", types[:3] + types[-1:]
    verifies = [e for e in events if e["type"] == "verify"]
    done = events[-1]
    produced = 0
    for v in verifies:
        drafts = [e for e in events if e["type"] == "draft" and e["step"] == v["step"]]
        assert len(drafts) == v["drafted"] == len(v["results"]), v
        statuses = [r["status"] for r in v["results"]]
        a = v["accepted"]
        assert statuses[:a] == ["accepted"] * a
        if a < v["drafted"]:
            assert statuses[a] == "rejected" and all(s == "discarded" for s in statuses[a + 1:])
        assert v["next"]["source"] in ("bonus", "resample", "base")
        produced += a + 1
    assert done["accepted"] == sum(v["accepted"] for v in verifies)
    assert done["tokens"] == produced <= steps + 0, (done, produced, steps)
    return done


async def main():
    params = RunParams.from_dict({"prompt": "Once upon a time", "steps": 60, "n_draft": 4, "seed": 7})
    sim = [e async for e in simulate(params, pace_s=0)]
    d = check(sim, 60)
    print(f"simulator ok: {d['tokens']} tokens, {d['steps']} steps, acceptance {d['accepted']/d['drafted']:.0%}")

    eng = Engine(EngineSettings())
    st = eng.settings.status()
    print("engine status:", st["engine_available"], st["reason"], [m["name"] for m in st["models"]])
    if st["engine_available"]:
        p = RunParams.from_dict({"prompt": "Once upon a time", "steps": 60, "n_draft": 4, "seed": 7,
                                 "base_model": "test-base.bin", "draft_model": "test-draft.bin"})
        ev = [e async for e in eng.stream(p)]
        d = check(ev, 60)
        print(f"engine ok: {d['tokens']} tokens, {d['steps']} steps, acceptance {d['accepted']/d['drafted']:.0%}, "
              f"{d['tokens']/d['gen_us']*1e6:.0f} tok/s")
        # rejects paths outside models/
        bad = RunParams.from_dict({"base_model": "../engine/run.c", "draft_model": "test-draft.bin"})
        try:
            eng.build_command(bad); raise SystemExit("path traversal not rejected")
        except Exception as e:
            print("rejects unknown model:", e)


asyncio.run(main())
