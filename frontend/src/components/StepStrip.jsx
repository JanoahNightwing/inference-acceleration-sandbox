/** One column per speculative step: how many of the drafted tokens the target model kept. */
export default function StepStrip({ steps, nDraft }) {
  const shown = steps.slice(-96);
  const h = Math.max(nDraft, 1);
  return (
    <div className="strip" role="img" aria-label={`${steps.length} speculative steps`}>
      {shown.map((s) => (
        <div key={s.step} className={`strip__col ${s.full ? "is-full" : ""}`} title={`Step ${s.step + 1}: ${s.accepted} of ${s.drafted} drafts accepted`}>
          {Array.from({ length: h }, (_, i) => (
            <span
              key={i}
              className={i < s.accepted ? "is-acc" : i < s.drafted ? "is-rej" : "is-none"}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
