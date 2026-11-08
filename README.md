# Inference Acceleration Sandbox

A live visualizer for speculative decoding. A small draft model guesses several tokens ahead, and a larger
target model checks all of them in one batched pass. The UI shows every step:

- **Draft guesses** pop in instantly in light blue.
- **A verification sweep** runs across them. Accepted guesses turn green. A rejected guess flashes red, the
  guesses after it drop out, and the target model's own token swaps in.
- **A speedometer** shows the engine's measured tokens per second against a notch for the target model
  decoding alone. Streaks of fully accepted steps push the needle past it and light the gauge up.

The inference engine in `engine/` is a minimal C implementation of speculative sampling built on Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)

```
React (Vite)  ──WebSocket──▶  FastAPI  ──subprocess, JSON lines──▶  speculative_decoding.c  -j 1
  token animation              /ws/generate                          draft / verify events
  speedometer, step strip      /api/status                           with p, q and timings
```

## Quick start

```bash
# 1. Engine
cd engine && make && cd ..

# 2. Checkpoints: stories15M drafts for stories42M (or stories110M)
cd models
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
cd ..

# 3. Backend (port 8000)
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 4. Frontend (port 5173, proxies /api and /ws to :8000)
cd frontend && npm install && npm run dev
```

For a single-process deployment, run `npm run build` in `frontend/`. FastAPI then serves `frontend/dist` at `/`.

Without checkpoints, the backend falls back to a **simulator** that emits the same event stream, so the UI
can be demoed anywhere. Simulated runs are labelled in the header. The simulator models a memory-bound
accelerator, where a draft token costs about 1/8 of a target forward.

## What changed in the C engine

`engine/` contains `speculative_decoding.c` with two additions. `patches/0001-json-event-stream.patch` applies
the same change to the upstream repo. The default colored terminal output is unchanged.

| Flag | Meaning |
|---|---|
| `-k <int>` | Draft tokens per speculative step (default 4, as before) |
| `-j 1` | Emit one JSON object per line instead of colored text |

Event stream (`-j 1`):

```jsonc
{"type":"start", "base":{...config}, "draft":{...config}, "n_draft":4, "steps":200, "temperature":1, "topp":0.9, "seed":42}
{"type":"prompt", "n_tokens":5, "pieces":["Once"," upon"," a"," time"], "prefill_us":2100.4}
{"type":"draft", "step":0, "i":0, "id":29892, "text":",", "q":0.91, "t_us":154.9}      // one per drafted token
{"type":"verify", "step":0, "pos":5, "drafted":4, "accepted":2,
 "results":[{"i":0,"status":"accepted","p":0.95,"q":0.91}, {"i":1,"status":"accepted",...},
            {"i":2,"status":"rejected","p":0.02,"q":0.40}, {"i":3,"status":"discarded"}],
 "next":{"id":763,"text":" there","source":"resample"},   // "bonus" when every draft was accepted
 "draft_us":410.2, "verify_us":1630.5, "base_token_us":980.1, "step_us":3300.8}
{"type":"done", "steps":61, "tokens":200, "drafted":240, "accepted":139, "gen_us":..., "total_us":...}
```

The speedometer reading for a step is `(accepted + 1) / step_us`. The "target alone" baseline is
`1 / base_token_us`: the measured cost of one single-token forward of the target model in the same run.

## Layout

```
engine/       speculative_decoding.c (+ run.c from llama2.c), tokenizer.bin, Makefile
backend/      FastAPI app: engine runner, simulator, WebSocket endpoint, stream tests
frontend/     React app: playback engine, token stream, speedometer, step strip
scripts/      make_test_models.py writes tiny random-weight checkpoints for smoke tests
patches/      the engine change as a patch against upstream speculative_decoding.c
```

## Tests

```bash
python scripts/make_test_models.py models   # tiny random checkpoints (text output is gibberish)
cd backend && python -m tests.test_streams  # checks both event sources produce consistent streams
```

## Notes

- Measured speedups on CPU depend on how cheap the batched verify pass is relative to single-token
  decoding. `run.c` processes a batch of tokens with a per-token loop, so CPU gains are modest. The
  sandbox reports what the engine measures, including steps where speculation costs time.
- `sample()` in `run.c` divides probabilities by the temperature without renormalising, so the draft `q`
  values are scaled when `temperature != 1`. Use temperature 1.0 for faithful acceptance statistics.
- Generation length is capped by the draft model's `seq_len`, a known limitation of the engine.

## License

MIT. `engine/run.c` keeps the llama2.c copyright notice (Andrej Karpathy, MIT).
