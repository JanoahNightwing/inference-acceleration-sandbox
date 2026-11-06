import { useCallback, useRef, useState } from "react";

// Animation timings in ms at 1x playback. The engine runs far faster than this;
// playback is a slowed-down replay, while the speedometer shows the engine's measured speed.
const T = { draft: 75, scan: 115, settle: 140, collapse: 260, swapIn: 60, gap: 90 };

const EMPTY_METRICS = {
  tps: 0, baselineTps: 0, peakTps: 0, nDraft: 4,
  tokens: 0, drafted: 0, accepted: 0, steps: 0,
  genUs: 0, baselineUs: 0, streak: 0, bestStreak: 0,
};

/**
 * Consumes engine events (start / prompt / draft / verify / done / error) and replays them
 * as token-level animation state for the UI.
 */
export function usePlayback() {
  const [tokens, setTokens] = useState([]);
  const [metrics, setMetrics] = useState(EMPTY_METRICS);
  const [steps, setSteps] = useState([]);
  const [run, setRun] = useState(null); // start event + phase
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState(null);

  const queue = useRef([]);
  const looping = useRef(false);
  const generation = useRef(0);
  const speed = useRef(1);
  const m = useRef(EMPTY_METRICS);
  const streamDone = useRef(false);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms / speed.current));

  const patch = (key, fields) =>
    setTokens((ts) => ts.map((t) => (t.key === key ? { ...t, ...fields } : t)));

  const handle = useCallback(async (ev, gen) => {
    const alive = () => gen === generation.current;
    switch (ev.type) {
      case "start":
        m.current = { ...EMPTY_METRICS, nDraft: ev.n_draft };
        setMetrics(m.current);
        setRun(ev);
        return;
      case "prompt":
        setTokens(ev.pieces.map((text, i) => ({ key: `p${i}`, text, kind: "prompt" })));
        return;
      case "draft":
        setTokens((ts) => [...ts, {
          key: `s${ev.step}d${ev.i}`, text: ev.text, kind: "draft", step: ev.step, i: ev.i, q: ev.q,
        }]);
        await sleep(T.draft);
        return;
      case "verify": {
        // sweep: the target model checks each draft token in order
        for (const r of ev.results) {
          if (!alive()) return;
          const key = `s${ev.step}d${r.i}`;
          if (r.status === "discarded") { patch(key, { kind: "discarded" }); continue; } // never checked
          patch(key, { scanning: true });
          await sleep(T.scan);
          patch(key, { scanning: false, kind: r.status, p: r.p, q: r.q });
          if (r.status === "rejected") await sleep(T.settle);
        }
        if (!alive()) return;
        const dropped = ev.results.filter((r) => r.status !== "accepted").map((r) => `s${ev.step}d${r.i}`);
        if (dropped.length) {
          setTokens((ts) => ts.map((t) => (dropped.includes(t.key) ? { ...t, leaving: true } : t)));
          await sleep(T.collapse);
          if (!alive()) return;
          setTokens((ts) => ts.filter((t) => !dropped.includes(t.key)));
        }
        setTokens((ts) => [...ts, {
          key: `s${ev.step}n`, text: ev.next.text, kind: "target", source: ev.next.source, step: ev.step,
          replaced: ev.results.find((r) => r.status === "rejected") ? true : false,
        }]);

        // metrics: measured engine speed for this step vs the target model decoding alone
        const c = m.current;
        const produced = ev.accepted + 1;
        const stepTps = (produced / ev.step_us) * 1e6;
        const baseTps = 1e6 / ev.base_token_us;
        const full = ev.drafted > 0 && ev.accepted === ev.drafted;
        const streak = full ? c.streak + 1 : 0;
        m.current = {
          ...c,
          tps: c.steps === 0 ? stepTps : c.tps * 0.35 + stepTps * 0.65,
          baselineTps: c.baselineTps ? c.baselineTps * 0.8 + baseTps * 0.2 : baseTps,
          peakTps: Math.max(c.peakTps, stepTps),
          tokens: c.tokens + produced,
          drafted: c.drafted + ev.drafted,
          accepted: c.accepted + ev.accepted,
          steps: c.steps + 1,
          genUs: c.genUs + ev.step_us,
          baselineUs: c.baselineUs + produced * ev.base_token_us,
          streak,
          bestStreak: Math.max(c.bestStreak, streak),
        };
        setMetrics(m.current);
        setSteps((s) => [...s, { step: ev.step, drafted: ev.drafted, accepted: ev.accepted, tps: stepTps, full }]);
        await sleep(T.gap);
        return;
      }
      case "done":
        setPhase("done");
        return;
      case "error":
        setError(ev.message);
        setPhase("error");
        return;
      default:
    }
  }, []);

  const loop = useCallback(async () => {
    if (looping.current) return;
    looping.current = true;
    const gen = generation.current;
    while (queue.current.length && gen === generation.current) {
      await handle(queue.current.shift(), gen);
    }
    looping.current = false;
    if (gen !== generation.current) return;
    if (queue.current.length) loop();
    else if (streamDone.current) setPhase((p) => (p === "running" ? "done" : p));
  }, [handle]);

  const push = useCallback((ev) => {
    if (ev.type === "done" || ev.type === "error" || ev.type === "stopped") streamDone.current = true;
    if (ev.type === "stopped") return;
    queue.current.push(ev);
    loop();
  }, [loop]);

  const reset = useCallback(() => {
    generation.current += 1;
    queue.current = [];
    looping.current = false;
    streamDone.current = false;
    m.current = EMPTY_METRICS;
    setTokens([]); setSteps([]); setMetrics(EMPTY_METRICS); setRun(null); setError(null);
    setPhase("running");
  }, []);

  const halt = useCallback(() => {
    generation.current += 1;
    queue.current = [];
    looping.current = false;
    setTokens((ts) => ts.filter((t) => !["draft", "rejected", "discarded"].includes(t.kind)).map((t) => ({ ...t, scanning: false })));
    setPhase((p) => (p === "running" ? "done" : p));
  }, []);

  const setSpeed = useCallback((v) => { speed.current = v; }, []);

  return { tokens, metrics, steps, run, phase, error, push, reset, halt, setSpeed, setError, setPhase };
}
