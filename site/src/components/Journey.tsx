import { createContext, useContext, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, CircleDot, CircleSlash, MinusCircle } from "lucide-react";
import { expName, journey, type JourneyStep } from "../data/content";
import { exp, OUTCOME_STYLE, PROTOCOL_LABEL } from "../data/registry";
import { Section } from "./Layout";
import { AXIS, GRID, LEGEND, SERIES_COLORS, TOOLTIP } from "./charts";

// Shared selection so the journey and findings can open an experiment in the explorer.
export const SelectionContext = createContext<{ selected: string; select: (id: string) => void }>({ selected: "", select: () => {} });
export const useSelection = () => useContext(SelectionContext);

const prettyKey = (k: string) => k.replace("gender_", "same-gender ").replace("standard_", "standard ").replace(/^gender$/, "same gender").replace(/_/g, " ");

function StepChart({ step }: { step: JourneyStep }) {
  const c = step.chart;
  if (!c) return null;
  const e = exp(c.id);
  if (c.id === "learning_curve") {
    return (
      <figure className="detail-visual" aria-label="Learning curve: EER versus number of training identities">
        <div className="chart" style={{ height: 240 }}>
          <ResponsiveContainer>
            <LineChart data={e.points} margin={{ top: 5, right: 14, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="identities" scale="log" type="number" domain={[18, 60]} ticks={[20, 28, 42, 56]} tick={AXIS} />
              <YAxis domain={["auto", "auto"]} tick={AXIS} unit="%" />
              <Tooltip contentStyle={TOOLTIP} />
              <Legend formatter={prettyKey} wrapperStyle={LEGEND} />
              {c.keys.map((k, i) => <Line key={k} dataKey={k} stroke={SERIES_COLORS[i]} strokeWidth={2} dot isAnimationActive={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <figcaption>Training identities per fold (log scale) vs EER · {PROTOCOL_LABEL[e.protocol]}.</figcaption>
      </figure>
    );
  }
  const table = (e[c.field ?? "systems"] as Record<string, Record<string, number>>) ?? {};
  const data = c.keys.map((k) => ({ key: prettyKey(k), ...Object.fromEntries(c.systems.map((s) => [s, table[s]?.[k]])) }));
  return (
    <figure className="detail-visual" aria-label={`EER comparison: ${c.systems.join(", ")}`}>
      <div className="chart" style={{ height: 250 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 5, right: 14, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="key" tick={AXIS} />
            <YAxis tick={AXIS} unit="%" domain={[0, "auto"]} />
            <Tooltip contentStyle={TOOLTIP} formatter={(v: number) => `${v.toFixed(2)}%`} />
            <Legend wrapperStyle={LEGEND} formatter={(v: string) => (v.length > 38 ? `${v.slice(0, 36)}…` : v)} />
            {c.systems.map((s, i) => <Bar key={s} dataKey={s} fill={SERIES_COLORS[i]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption>EER (%), lower is better · {PROTOCOL_LABEL[e.protocol]}.</figcaption>
    </figure>
  );
}

const SUPPORT = {
  yes: { icon: CheckCircle2, label: "Hypothesis supported", short: "supported", cls: "yes" },
  partly: { icon: CircleDot, label: "Partly supported", short: "partly", cls: "partly" },
  no: { icon: CircleSlash, label: "Hypothesis not supported", short: "not supported", cls: "no" },
  "n/a": { icon: MinusCircle, label: "Problem statement", short: "context", cls: "na" },
};

export function Journey() {
  const [i, setI] = useState(0);
  const step = journey[i];
  const { select } = useSelection();
  const S = SUPPORT[step.supported];
  const n2 = (k: number) => String(k + 1).padStart(2, "0");
  return (
    <Section id="journey" n="04" band eyebrow="RESEARCH JOURNEY" title="Problem → hypotheses → experiments → discoveries → final system"
      intro="Eight decisions, including the ones that failed. Each step states what we believed, what we changed, how it was tested, what happened and what we did next.">
      <div className="explorer">
        <div className="explorer-heading">
          <div>
            <p className="kicker">EXPLORE THE JOURNEY</p>
            <h3>Eight steps, from question to final system</h3>
            <p>Select a step, or use Next to walk through the project in order.</p>
          </div>
          <span className="explorer-progress">{n2(i)} — {n2(journey.length - 1)}</span>
        </div>
        <ol className="explorer-track four" role="tablist" aria-label="Research steps">
          {journey.map((s, k) => {
            const st = SUPPORT[s.supported];
            const on = k === i;
            return (
              <li key={s.id}>
                <button type="button" role="tab" aria-selected={on} className={`explorer-card compact ${on ? "is-active" : ""}`} onClick={() => setI(k)}>
                  <span className="card-top"><span className="card-number">{n2(k)}</span><st.icon aria-hidden="true" /></span>
                  <strong>{s.stage}</strong>
                  <span className="card-sub">{s.title}</span>
                  <span className={`card-action status ${st.cls}`}>{st.short}<span aria-hidden="true">{on ? "✓" : "↗"}</span></span>
                </button>
              </li>
            );
          })}
        </ol>

        <div className="explorer-panel" role="tabpanel" aria-labelledby="step-title" aria-live="polite">
          <div className="panel-header">
            <div>
              <span className="detail-number">STEP {n2(i)} · {step.stage}</span>
              <h4 id="step-title">{step.title}</h4>
              <p className={`status ${S.cls}`}><S.icon size={14} aria-hidden="true" /> {S.label}</p>
            </div>
            <div className="chips" style={{ flexWrap: "nowrap" }}>
              <button type="button" className="detail-close" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0} aria-label="Previous step">‹</button>
              <button type="button" className="detail-close" onClick={() => setI(Math.min(journey.length - 1, i + 1))} disabled={i === journey.length - 1} aria-label="Next step">›</button>
            </div>
          </div>
          <div key={step.id} className="panel-content is-changing">
            <div className="detail-copy">
              <dl>
                {([["What we believed", step.believed, ""], ["What we changed", step.changed, ""], ["How we tested it", step.experiment, ""], ["What happened", step.result, "result"], ["Decision", step.decision, "decision"]] as const).map(([k, v, cls]) => (
                  <div key={k}><dt>{k}</dt><dd className={cls}>{v}</dd></div>
                ))}
              </dl>
              <div className="buttons" style={{ justifyContent: "flex-start", marginBottom: 0 }}>
                <button type="button" className="button small" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>← Previous</button>
                <button type="button" className="button small primary" onClick={() => setI(Math.min(journey.length - 1, i + 1))} disabled={i === journey.length - 1}>Next step →</button>
              </div>
            </div>
            <div>
              <StepChart step={step} />
              <p className="lane-label" style={{ marginTop: 20 }}>Experiments behind this step</p>
              <ul className="link-list">
                {step.refs.map((r) => {
                  const e = exp(r);
                  return (
                    <li key={r}>
                      <a href="#experiments" onClick={() => select(r)} style={{ fontWeight: 500, fontSize: "0.88rem", padding: "10px 14px" }}>
                        <span style={{ color: "#30465d" }}>{expName(e)}</span>
                        <i className={`chip ${OUTCOME_STYLE[e.outcome]}`} style={{ fontStyle: "normal" }}>{e.outcome}</i>
                      </a>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
