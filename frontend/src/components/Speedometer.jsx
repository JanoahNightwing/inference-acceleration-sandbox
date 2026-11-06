import { useEffect, useRef, useState } from "react";

const START = -120; // degrees clockwise from 12 o'clock; the dial spans -120..+120
const SWEEP = 240;
const CX = 160, CY = 160, R = 124;

function polar(deg, r = R) {
  const a = ((deg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

function arc(from, to, r = R) {
  const [x1, y1] = polar(from, r);
  const [x2, y2] = polar(to, r);
  const large = to - from > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

function niceMax(v) {
  if (v <= 0) return 100;
  const mag = 10 ** Math.floor(Math.log10(v));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (step * mag >= v) return step * mag;
  }
  return 10 * mag;
}

function fmt(v) {
  if (v >= 10000) return `${Math.round(v / 1000)}k`;
  if (v >= 1000) return `${(v / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${Math.round(v)}`;
}

/** Needle follows `value` with a damped spring, so streaks visibly kick it upward. */
function useSpring(target) {
  const [v, setV] = useState(target);
  const state = useRef({ x: target, vel: 0, target });
  state.current.target = target;
  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { state.current.x = target; setV(target); return; }
    let raf, last = performance.now();
    const tick = (now) => {
      const s = state.current;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const force = (s.target - s.x) * 90 - s.vel * 13;
      s.vel += force * dt;
      s.x += s.vel * dt;
      setV(s.x);
      if (Math.abs(s.target - s.x) > 0.05 || Math.abs(s.vel) > 0.05) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return v;
}

export default function Speedometer({ tps, baselineTps, peakTps, nDraft, streak }) {
  // scale: room for the best case (every draft accepted) without rescaling mid-run
  const ceiling = useRef(0);
  const want = niceMax(Math.max(baselineTps * (nDraft + 1) * 0.9, peakTps * 1.05, 10));
  if (want > ceiling.current || baselineTps === 0) ceiling.current = want;
  const max = ceiling.current;

  const shown = useSpring(Math.min(tps, max));
  const angle = (val) => START + (Math.max(0, Math.min(val, max)) / max) * SWEEP;
  const needle = angle(shown);
  const base = angle(baselineTps);
  const speedup = baselineTps > 0 ? tps / baselineTps : 0;
  const hot = streak >= 2;
  const ticks = Array.from({ length: 11 }, (_, i) => (i / 10) * max);

  return (
    <figure className={`gauge ${hot ? "gauge--hot" : ""}`} aria-label={`Engine speed ${Math.round(tps)} tokens per second`}>
      <svg viewBox="-36 4 392 266" role="img">
        <defs>
          <linearGradient id="fast" x1="0" x2="1">
            <stop offset="0" stopColor="var(--accept)" />
            <stop offset="1" stopColor="var(--draft)" />
          </linearGradient>
        </defs>
        <path d={arc(START, START + SWEEP)} className="gauge__track" />
        {baselineTps > 0 && shown > 0 && (
          <path
            d={arc(START, Math.max(START + 0.5, needle))}
            className={`gauge__fill ${shown >= baselineTps ? "is-faster" : "is-slower"}`}
          />
        )}
        {ticks.map((t, i) => {
          const d = angle(t);
          const [x1, y1] = polar(d, R - 16);
          const [x2, y2] = polar(d, R - (i % 2 ? 22 : 28));
          const [lx, ly] = polar(d, R - 44);
          return (
            <g key={i}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} className="gauge__tick" />
              {i % 2 === 0 && <text x={lx} y={ly} className="gauge__ticklabel">{fmt(t)}</text>}
            </g>
          );
        })}
        {baselineTps > 0 && (
          <g className="gauge__baseline">
            <line x1={polar(base, R + 12)[0]} y1={polar(base, R + 12)[1]} x2={polar(base, R - 12)[0]} y2={polar(base, R - 12)[1]} />
            <text
              x={polar(base, R + 18)[0]} y={polar(base, R + 18)[1]}
              textAnchor={polar(base)[0] < CX - 8 ? "end" : polar(base)[0] > CX + 8 ? "start" : "middle"}
            >target alone</text>
          </g>
        )}
        <g style={{ transform: `rotate(${needle}deg)`, transformOrigin: `${CX}px ${CY}px` }}>
          <line x1={CX} y1={CY + 14} x2={CX} y2={CY - R + 20} className="gauge__needle" />
        </g>
        <circle cx={CX} cy={CY} r="7" className="gauge__hub" />
        <text x={CX} y={CY + 76} className="gauge__value">{fmt(tps)}</text>
        <text x={CX} y={CY + 98} className="gauge__unit">tokens / s</text>
      </svg>
      <figcaption>
        <span className={`gauge__speedup ${speedup >= 1 ? "is-faster" : "is-slower"}`}>
          {speedup ? `${speedup.toFixed(2)}×` : "—"}
        </span>{" "}
        <span className="gauge__caption">the target model decoding on its own</span>
        {hot && <span className="gauge__streak">{streak} full steps in a row</span>}
      </figcaption>
    </figure>
  );
}
