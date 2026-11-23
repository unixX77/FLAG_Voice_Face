import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { experiments, fmt, fmtCI, fmtDelta, OUTCOME_STYLE, PROTOCOL_LABEL, sourceUrl, type Delta, type Experiment, type Metrics } from "../data/registry";
import { KEY_EXPERIMENTS, expName } from "../data/content";
import { Section } from "./Layout";
import { useSelection } from "./Journey";
import { AXIS_SMALL, GRID, LEGEND, SERIES_COLORS, TOOLTIP } from "./charts";

const KEY_VIEW = KEY_EXPERIMENTS.map((k) => k.id);
const label = (k: string) => k.replace(/^gender_/, "same-gender ").replace(/^standard_/, "standard ").replace(/^gender$/, "same gender").replace(/_/g, " ");
const uniq = (xs: string[]) => Array.from(new Set(xs)).sort();
const deltaCls = (d?: Delta) => (d?.ci95 && d.ci95[1] < 0 ? "better" : d?.ci95 && d.ci95[0] > 0 ? "worse" : "");

function SystemsChart({ systems }: { systems: Record<string, Metrics> }) {
  const names = Object.keys(systems);
  const keys = uniq(names.flatMap((n) => Object.keys(systems[n]))).filter((k) => !["mean"].includes(k) || names.length < 6);
  const ordered = ["standard", "gender", "mean", "standard_English", "standard_Urdu", "standard_German", "gender_English", "gender_Urdu", "gender_German"].filter((k) => keys.includes(k));
  const data = (ordered.length ? ordered : keys).map((k) => ({ key: label(k), ...Object.fromEntries(names.map((n) => [n, systems[n][k]])) }));
  return (
    <div className="chart" style={{ height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="key" tick={AXIS_SMALL} interval={0} />
          <YAxis tick={AXIS_SMALL} unit="%" />
          <Tooltip contentStyle={TOOLTIP} formatter={(v: number) => (typeof v === "number" ? `${v.toFixed(2)}%` : "n/a")} />
          <Legend wrapperStyle={LEGEND} formatter={(v: string) => (v.length > 34 ? `${v.slice(0, 32)}…` : v)} />
          {names.map((n, i) => <Bar key={n} dataKey={n} fill={SERIES_COLORS[i % SERIES_COLORS.length]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function FoldChart({ perFold, track }: { perFold: Record<string, Record<string, number[]>>; track: "standard" | "gender" }) {
  const names = Object.keys(perFold);
  const data = [0, 1, 2, 3, 4].map((f) => ({ fold: `fold ${f}`, ...Object.fromEntries(names.map((n) => [n, perFold[n][track][f]])) }));
  return (
    <div className="chart" style={{ height: 200 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="fold" tick={AXIS_SMALL} />
          <YAxis tick={AXIS_SMALL} unit="%" domain={["auto", "auto"]} />
          <Tooltip contentStyle={TOOLTIP} formatter={(v: number) => `${v.toFixed(2)}%`} />
          <Legend wrapperStyle={LEGEND} formatter={(v: string) => (v.length > 34 ? `${v.slice(0, 32)}…` : v)} />
          {names.map((n, i) => <Line key={n} dataKey={n} stroke={SERIES_COLORS[i]} strokeWidth={2} dot isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function KeyValueTable({ rows, caption }: { rows: Record<string, Metrics | number | null>; caption: string }) {
  const entries = Object.entries(rows);
  const cols = uniq(entries.flatMap(([, v]) => (v && typeof v === "object" ? Object.keys(v) : [])));
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead><tr><th>Item</th>{cols.length ? cols.map((c) => <th key={c} className="num">{label(c)}</th>) : <th className="num">Value</th>}</tr></thead>
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td>{label(k)}</td>
              {v && typeof v === "object" ? cols.map((c) => <td key={c} className="num">{typeof v[c] === "number" ? (v[c] <= 1 && !c.includes("EER") && c.match(/R1|top|precision|ari|hungarian/i) ? v[c].toFixed(3) : fmt(v[c])) : "—"}</td>)
                : <td className="num">{typeof v === "number" ? v.toFixed(3) : "n/a"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Detail({ e }: { e: Experiment }) {
  const [track, setTrack] = useState<"standard" | "gender">("gender");
  const deltaKeys = e.deltas ? uniq(Object.values(e.deltas).flatMap((x) => Object.keys(x))) : [];
  return (
    <article className="explorer-panel detail" style={{ marginTop: 0 }} aria-live="polite">
      <div className="chips">
        <span className={`chip ${OUTCOME_STYLE[e.outcome]}`}>{e.outcome}</span>
        <span className="chip chip-neutral">{PROTOCOL_LABEL[e.protocol]}</span>
        {e.seeds && <span className="chip chip-neutral">{e.seeds.length === 1 ? `seed ${e.seeds[0]} only` : `${e.seeds.length} seeds × 5 folds`}</span>}
        <span className="note small">{e.phase}</span>
      </div>
      <h4>{expName(e)}</h4>
      <p className="qa"><b>Question</b>{e.question}</p>
      {e.hypothesis && <p className="qa"><b>Hypothesis</b>{e.hypothesis}</p>}
      <p className="interp">{e.interpretation}</p>

      {e.systems && <><p className="sub-title">Results</p><SystemsChart systems={e.systems} /></>}
      {e.systems_urdu && <><p className="sub-title">Urdu holdout</p><SystemsChart systems={e.systems_urdu} /></>}
      {e.systems_german && <><p className="sub-title">German holdout</p><SystemsChart systems={e.systems_german} /></>}
      {e.cells && <><p className="sub-title">Official cells</p><SystemsChart systems={e.cells} /></>}
      {e.points && <KeyValueTable caption="EER (%) by number of training identities" rows={Object.fromEntries(e.points.map((p) => [`${p.identities} identities`, { standard: p.standard, gender: p.gender, mean: p.mean }]))} />}
      {e.unimodal && <KeyValueTable caption="Unimodal EER (%) / retrieval R@1 on unseen people" rows={e.unimodal} />}
      {e.recovery && <KeyValueTable caption="Association recovery (fractions; 1.0 = perfect)" rows={e.recovery} />}

      {e.comparison && (
        <div className="table-wrap">
          <table>
            <caption>Paired difference {e.comparison.b} − {e.comparison.a} (EER points; negative = better), 95% identity-bootstrap interval.</caption>
            <thead><tr><th>Track</th><th className="num">Δ</th><th className="num">95% CI</th><th className="num">P(B better)</th></tr></thead>
            <tbody>
              {Object.entries(e.comparison.delta).map(([k, d]) => (
                <tr key={k}>
                  <td>{label(k)}</td>
                  <td className={`num ${deltaCls(d)}`}>{fmtDelta(d.value)}</td>
                  <td className="num">{fmtCI(d.ci95) || "—"}</td>
                  <td className="num">{d.p_b_better !== undefined ? d.p_b_better.toFixed(2) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {e.deltas && (
        <div className="table-wrap">
          <table>
            <caption>Paired differences (EER points; negative = better), 95% identity-bootstrap intervals.</caption>
            <thead><tr><th>Comparison</th>{deltaKeys.map((k) => <th key={k} className="num">{label(k)}</th>)}</tr></thead>
            <tbody>
              {Object.entries(e.deltas).map(([pair, ds]) => (
                <tr key={pair}>
                  <td>{pair}</td>
                  {deltaKeys.map((k) => (
                    <td key={k} className={`num ${deltaCls(ds[k])}`}>{ds[k] ? <>{fmtDelta(ds[k].value)}<span className="ci">{fmtCI(ds[k].ci95)}</span></> : "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {e.per_fold && (
        <>
          <div className="toolbar" style={{ marginTop: 22 }}>
            <p className="sub-title" style={{ margin: 0 }}>Fold consistency</p>
            <div className="seg">{(["standard", "gender"] as const).map((t) => <button key={t} aria-pressed={track === t} onClick={() => setTrack(t)}>{t === "gender" ? "Same gender" : "Standard"}</button>)}</div>
          </div>
          <FoldChart perFold={e.per_fold} track={track} />
        </>
      )}
      <p className="lane-label" style={{ marginTop: 22 }}>Sources</p>
      <ul className="sources">
        {e.sources.map((s) => (
          <li key={s.path + (s.lines ?? "")}>
            <a href={sourceUrl(s.path, s.lines)} target="_blank" rel="noreferrer">{s.path}{s.lines ? ` (lines ${s.lines})` : ""} ↗</a>
            {s.type === "progress_log" && <span className="note"> · from the research log</span>}
          </li>
        ))}
      </ul>
    </article>
  );
}

export function ExperimentExplorer() {
  const { selected, select } = useSelection();
  // Key experiments only; an experiment opened from elsewhere on the page (Journey, Findings) is appended as an extra card.
  const ids = KEY_VIEW.includes(selected) || !experiments.some((e) => e.id === selected) ? KEY_VIEW : [...KEY_VIEW, selected];
  const list = ids.map((id) => experiments.find((e) => e.id === id)!).filter(Boolean);
  const current = list.find((e) => e.id === selected) ?? list[0];
  return (
    <Section id="experiments" n="05" eyebrow="EXPERIMENT EXPLORER" title="Key experiments, every number traceable"
      intro={<>The comparisons that changed the project's direction. Select an experiment to see its question, protocol, result and source files. EER in %, lower is better; differences are B − A in EER points.</>}>
      <ol className="explorer-track" role="tablist" aria-label="Key experiments" style={{ marginTop: 30 }}>
        {list.map((e, k) => {
          const on = e.id === current.id;
          return (
            <li key={e.id}>
              <button type="button" role="tab" aria-selected={on} className={`explorer-card compact ${on ? "is-active" : ""}`} onClick={() => select(e.id)}>
                <span className="card-top"><span className="card-number">{String(k + 1).padStart(2, "0")}</span><span className={`chip ${OUTCOME_STYLE[e.outcome]}`}>{e.outcome}</span></span>
                <strong>{expName(e)}</strong>
                <span className="card-sub">{KEY_EXPERIMENTS.find((x) => x.id === e.id)?.desc ?? PROTOCOL_LABEL[e.protocol]}</span>
                <span className="card-action">{on ? "Selected" : "View experiment"}<span aria-hidden="true">{on ? "✓" : "↗"}</span></span>
              </button>
            </li>
          );
        })}
      </ol>
      <div style={{ marginTop: 18 }}>
        <Detail e={current} />
      </div>
    </Section>
  );
}
