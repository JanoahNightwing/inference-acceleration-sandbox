import { useCallback, useEffect, useRef, useState } from "react";
import { usePlayback } from "./hooks/usePlayback.js";
import Speedometer from "./components/Speedometer.jsx";
import TokenStream from "./components/TokenStream.jsx";
import StepStrip from "./components/StepStrip.jsx";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/generate`;
}

function params(n) {
  return Number.isFinite(n) ? n : 0;
}

function paramLabel(cfg) {
  if (!cfg) return "";
  // rough llama2 parameter count: embeddings + attention + feed-forward
  const kv = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
  const perLayer = cfg.dim * (2 * cfg.dim + 2 * kv) + 3 * cfg.dim * cfg.hidden_dim;
  const n = cfg.vocab_size * cfg.dim + cfg.n_layers * perLayer;
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${Math.round(n / 1e6)}M` : `${Math.round(n / 1e3)}K`;
}

export default function App() {
  const pb = usePlayback();
  const [status, setStatus] = useState(null);
  const [prompt, setPrompt] = useState("Once upon a time");
  const [nDraft, setNDraft] = useState(4);
  const [temperature, setTemperature] = useState(1.0);
  const [steps, setSteps] = useState(160);
  const [seed, setSeed] = useState("");
  const [source, setSource] = useState("auto");
  const [baseModel, setBaseModel] = useState("");
  const [draftModel, setDraftModel] = useState("");
  const [speed, setSpeed] = useState(1);
  const ws = useRef(null);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then((s) => {
        setStatus(s);
        setBaseModel(s.default_base || "");
        setDraftModel(s.default_draft || "");
        setSource(s.engine_available ? "engine" : "simulated");
      })
      .catch(() => setStatus({ engine_available: false, reason: "The backend is not reachable on /api.", models: [] }));
  }, []);

  const { setSpeed: setPlaybackSpeed } = pb;
  useEffect(() => setPlaybackSpeed(speed), [speed, setPlaybackSpeed]);

  const socket = useCallback(() => new Promise((resolve, reject) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return resolve(ws.current);
    const s = new WebSocket(wsUrl());
    s.onopen = () => { ws.current = s; resolve(s); };
    s.onerror = () => reject(new Error("Could not open the generation socket. Is the backend running on port 8000?"));
    s.onmessage = (e) => pb.push(JSON.parse(e.data));
    s.onclose = () => { if (ws.current === s) ws.current = null; };
  }), [pb]);

  const generate = async (e) => {
    e?.preventDefault();
    pb.reset();
    try {
      const s = await socket();
      s.send(JSON.stringify({
        type: "run", source, prompt, n_draft: nDraft, temperature, steps,
        seed: seed === "" ? 0 : params(parseInt(seed, 10)),
        base_model: baseModel, draft_model: draftModel,
      }));
    } catch (err) {
      pb.setError(err.message);
      pb.setPhase("error");
    }
  };

  const stop = () => {
    ws.current?.send(JSON.stringify({ type: "stop" }));
    pb.halt();
  };

  const m = pb.metrics;
  const run = pb.run;
  const acceptance = m.drafted ? m.accepted / m.drafted : 0;
  const perStep = m.steps ? m.tokens / m.steps : 0;
  const savedMs = (m.baselineUs - m.genUs) / 1000;
  const engineOk = status?.engine_available;
  const running = pb.phase === "running";

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Inference Acceleration Sandbox</h1>
          <p className="top__sub">
            A small draft model guesses ahead. The large target model checks every guess in one pass and keeps
            the ones it agrees with.
          </p>
        </div>
        <p className={`runline ${run?.source === "simulated" ? "is-sim" : ""}`}>
          {run
            ? run.source === "simulated"
              ? "Simulated run with synthetic timings. Add checkpoints to models/ to run the C engine."
              : `speculative_decoding.c: ${run.base_name} (${paramLabel(run.base)}) verifying ${run.draft_name} (${paramLabel(run.draft)})`
            : status
              ? engineOk ? `Engine ready with ${status.models.length} checkpoints` : status.reason
              : "Connecting to the backend"}
        </p>
      </header>

      <form className="prompt" onSubmit={generate}>
        <label htmlFor="prompt" className="sr-only">Prompt</label>
        <input id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Start a story" maxLength={500} />
        {running
          ? <button type="button" className="btn btn--stop" onClick={stop}>Stop</button>
          : <button type="submit" className="btn">Generate</button>}
      </form>

      <main className="stage">
        <TokenStream tokens={pb.tokens} phase={pb.phase} />
        <aside className="panel">
          <Speedometer tps={m.tps} baselineTps={m.baselineTps} peakTps={m.peakTps} nDraft={m.nDraft} streak={m.streak} />
          <dl className="stats">
            <div><dt>Draft acceptance</dt><dd>{m.drafted ? `${Math.round(acceptance * 100)}%` : "—"}</dd></div>
            <div><dt>Tokens per pass</dt><dd>{m.steps ? perStep.toFixed(2) : "—"}</dd></div>
            <div><dt>Best streak</dt><dd>{m.steps ? m.bestStreak : "—"}</dd></div>
            <div>
              <dt>{savedMs >= 0 ? "Time saved" : "Time lost"}</dt>
              <dd>{m.steps ? `${Math.abs(savedMs).toFixed(savedMs > -10 && savedMs < 10 ? 1 : 0)} ms` : "—"}</dd>
            </div>
          </dl>
        </aside>
      </main>

      <section className="steps">
        <div className="steps__head">
          <h2>Speculative steps</h2>
          <p>Each column is one target pass. Filled cells are draft tokens the target model accepted.</p>
        </div>
        <StepStrip steps={pb.steps} nDraft={m.nDraft} />
      </section>

      {pb.error && <p className="error" role="alert">{pb.error}</p>}

      <section className="controls" aria-label="Settings">
        <fieldset className="seg">
          <legend>Backend</legend>
          <label className={!engineOk ? "is-disabled" : ""} title={engineOk ? "" : status?.reason || ""}>
            <input type="radio" name="src" value="engine" checked={source === "engine"} disabled={!engineOk} onChange={() => setSource("engine")} />
            C engine
          </label>
          <label>
            <input type="radio" name="src" value="simulated" checked={source === "simulated"} onChange={() => setSource("simulated")} />
            Simulated
          </label>
        </fieldset>

        {engineOk && source === "engine" && (
          <>
            <label className="field">
              <span>Target model</span>
              <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
                {status.models.map((mm) => <option key={mm.name} value={mm.name}>{mm.name} ({mm.size_mb} MB)</option>)}
              </select>
            </label>
            <label className="field">
              <span>Draft model</span>
              <select value={draftModel} onChange={(e) => setDraftModel(e.target.value)}>
                {status.models.map((mm) => <option key={mm.name} value={mm.name}>{mm.name} ({mm.size_mb} MB)</option>)}
              </select>
            </label>
          </>
        )}

        <label className="field">
          <span>Draft length <output>{nDraft}</output></span>
          <input type="range" min="0" max="8" value={nDraft} onChange={(e) => setNDraft(+e.target.value)} />
        </label>
        <label className="field">
          <span>Temperature <output>{temperature.toFixed(1)}</output></span>
          <input type="range" min="0" max="1.5" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
        </label>
        <label className="field">
          <span>Max tokens <output>{steps}</output></span>
          <input type="range" min="16" max="256" step="16" value={steps} onChange={(e) => setSteps(+e.target.value)} />
        </label>
        <label className="field field--narrow">
          <span>Seed</span>
          <input inputMode="numeric" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} />
        </label>
        <fieldset className="seg">
          <legend>Playback</legend>
          {SPEEDS.map((s) => (
            <label key={s}>
              <input type="radio" name="speed" checked={speed === s} onChange={() => setSpeed(s)} />
              {s}×
            </label>
          ))}
        </fieldset>
      </section>
    </div>
  );
}
