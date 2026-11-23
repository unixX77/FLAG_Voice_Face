import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { exp, fmt, metric, registry, REPO_URL } from "../data/registry";
import { Section } from "./Layout";
import { AXIS, GRID, TOOLTIP } from "./charts";

// Dataset facts verified in docs/publication/01_audit.md (manifest counts).
export const DATA = {
  sources: [
    { name: "FLAG official (MAV-Celeb v4)", ids: 70, langs: "English", note: "validation folds come only from here" },
    { name: "MAV-Celeb v1", ids: 64, langs: "English + Urdu", note: "" },
    { name: "MAV-Celeb v3", ids: 50, langs: "English + German", note: "" },
    { name: "VoxCeleb2 test", ids: 117, langs: "mostly English (language ID)", note: "" },
  ],
  externalRows: 29702, totalIds: 301, totalRows: 36187,
  externalByLanguage: [{ lang: "English", rows: 6990 }, { lang: "Urdu", rows: 12125 }, { lang: "German", rows: 1565 }, { lang: "VoxCeleb2 (unlabelled)", rows: 9022 }],
};

export function Hero() {
  const final = metric("final_c1x3_terms", "C1 x 3 (raw-mean ensemble) + terms", "mean");
  const b0 = metric("b0_terms", "B0 + terms", "mean");
  const fop = metric("codabench_eval_submissions", "FOP baseline (organisers)", "mean");
  return (
    <header id="top" className="hero">
      <div className="container center">
        <p className="eyebrow">FLAG 2027 · ICASSP 2027 GRAND CHALLENGE</p>
        <h1>Matching Faces and Voices Across Languages</h1>
        <p className="author">Kondra Varun Teja</p>
        <p className="hero-text">
          Cross-modal identity verification under unseen-language and same-gender conditions.
        </p>
        <div className="buttons" aria-label="Project links">
          <a className="button primary" href="#journey">Follow the research</a>
          <a className="button" href={REPO_URL} target="_blank" rel="noreferrer">View code ↗</a>
        </div>
        <div className="stats" aria-label="Main results">
          <div><b className="num">{fmt(final)}%</b><span>Final system (C1 × 3) · vs {fmt(b0)}% for B0</span></div>
          <div><b className="num">{fmt(fop)}%</b><span>Organisers' FOP baseline · official CodaBench evaluation phase</span></div>
          <div><b className="num">{DATA.totalIds}</b><span>training identities: 70 official + 231 external, {DATA.totalRows.toLocaleString("en-US")} face–voice pairs</span></div>
        </div>
      </div>
    </header>
  );
}

function EerExplorer() {
  const [track, setTrack] = useState<"standard" | "gender">("gender");
  const h = registry.score_histograms.tracks[track];
  const totals = useMemo(() => ({ g: h.genuine.reduce((a, b) => a + b, 0), i: h.impostor.reduce((a, b) => a + b, 0) }), [h]);
  const data = useMemo(() => h.genuine.map((c, k) => ({ x: (h.bins[k] + h.bins[k + 1]) / 2, genuine: c / totals.g, impostor: h.impostor[k] / totals.i })), [h, totals]);
  // Start at the bin where false accepts and false rejects are closest (the EER point).
  const eerIdx = useMemo(() => {
    let best = 0, gap = Infinity, frr = 0, far = totals.i;
    for (let k = 0; k < h.genuine.length; k++) {
      const d = Math.abs(far / totals.i - frr / totals.g);
      if (d < gap) { gap = d; best = k; }
      frr += h.genuine[k]; far -= h.impostor[k];
    }
    return best;
  }, [h, totals]);
  const [idxState, setIdx] = useState<number | null>(null);
  const idx = idxState ?? eerIdx;
  const thr = h.bins[idx];
  const frr = (100 * h.genuine.slice(0, idx).reduce((a, b) => a + b, 0)) / totals.g;
  const far = (100 * h.impostor.slice(idx).reduce((a, b) => a + b, 0)) / totals.i;
  return (
    <div className="explorer">
      <div className="explorer-heading">
        <div>
          <p className="kicker">INTERACTIVE · EQUAL ERROR RATE</p>
          <h3>Move the threshold, watch the errors trade off</h3>
          <p>Real cosine scores of B0 on unseen people ({registry.score_histograms.pooled_over}).</p>
        </div>
        <div className="seg" role="tablist" aria-label="Evaluation track">
          {(["standard", "gender"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={track === t} onClick={() => { setTrack(t); setIdx(null); }}>{t === "standard" ? "Standard" : "Same gender"}</button>
          ))}
        </div>
      </div>
      <div className="explorer-panel">
        <div className="chart" style={{ height: 240 }} aria-hidden="true">
          <ResponsiveContainer>
            <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" domain={[-0.2, 1]} tick={AXIS} tickFormatter={(v) => v.toFixed(1)} />
              <YAxis tick={AXIS} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip contentStyle={TOOLTIP} labelFormatter={(v) => `cosine ≈ ${Number(v).toFixed(2)}`}
                formatter={(v: number, n: string) => [`${(v * 100).toFixed(1)}%`, n === "genuine" ? "same person" : "different people"]} />
              <Area type="stepAfter" dataKey="impostor" stroke="#8c1515" fill="#8c1515" fillOpacity={0.12} isAnimationActive={false} />
              <Area type="stepAfter" dataKey="genuine" stroke="#1664c0" fill="#1664c0" fillOpacity={0.16} isAnimationActive={false} />
              <ReferenceLine x={thr} stroke="#b88316" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <ul className="legend" aria-hidden="true">
          <li><i style={{ background: "#1664c0" }} /> same person (genuine)</li>
          <li><i style={{ background: "#8c1515" }} /> different people (impostor)</li>
          <li><i style={{ background: "#b88316" }} /> threshold</li>
        </ul>
        <label className="field" htmlFor="thr" style={{ marginTop: 18 }}>Decision threshold: accept as “same person” if cosine ≥ <span className="num">{thr.toFixed(3)}</span></label>
        <input id="thr" type="range" min={20} max={74} value={idx} onChange={(e) => setIdx(Number(e.target.value))} />
        <div className="meters">
          <div><span>False accepts</span><b className="num bad">{far.toFixed(1)}%</b></div>
          <div><span>False rejects</span><b className="num">{frr.toFixed(1)}%</b></div>
          <div><span>EER (registry)</span><b className="num gold">{fmt(h.eer)}%</b></div>
        </div>
        <p className="note" style={{ marginTop: 12 }}>The EER is where false accepts equal false rejects. Rates here come from 0.025-wide histogram bins, so they approximate the exact EER computed from the trials.</p>
      </div>
    </div>
  );
}

export function Challenge() {
  const e = exp("codabench_eval_submissions");
  const key = "Eval #3: V2 + external data + terms";
  const cells = e.cells![key];
  const tiles = [
    { k: "standard_English", label: "STANDARD · HEARD LANGUAGE", t: "Standard · English", d: "Impostors of any gender; English is in the training data." },
    { k: "standard_Bangla", label: "STANDARD · UNHEARD LANGUAGE", t: "Standard · Bangla", d: "Impostors of any gender; Bangla is never heard in training." },
    { k: "gender_English", label: "SAME GENDER · HEARD LANGUAGE", t: "Same gender · English", d: "Every impostor shares the speaker's gender, so the easiest cue disappears." },
    { k: "gender_Bangla", label: "SAME GENDER · UNHEARD LANGUAGE", t: "Same gender · Bangla", d: "The hardest cell: same-gender impostors in a language never heard." },
  ];
  return (
    <Section id="challenge" n="01" eyebrow="THE CHALLENGE" title="Is this face the speaker of this voice?"
      intro={<>Each trial pairs one face image with one voice clip, and a system outputs a score. The challenge measures the <strong>Equal Error Rate</strong>: the error at the threshold where wrongly accepting impostors is as frequent as wrongly rejecting true pairs. A voice cannot determine a face, but both reveal shared traits such as sex, age and build.</>}>
      <div className="challenge-grid">
        {tiles.map((c, i) => (
          <article key={c.k} className="challenge-card">
            <span className="challenge-number">0{i + 1}</span>
            <div>
              <p className="challenge-label">{c.label}</p>
              <h3>{c.t}</h3>
              <p>{c.d}</p>
              <span className="big-number num">{fmt(cells[c.k])}%</span>
              <p className="note">official EER of our eval #3 submission</p>
            </div>
          </article>
        ))}
      </div>
      <p className="challenge-footnote">Official CodaBench evaluation-phase scores (mean {fmt(e.systems![key].mean)}%). Bangla is never used for training or tuning, as the challenge rules require.</p>
      <EerExplorer />
    </Section>
  );
}

export function DataMethod() {
  const max = Math.max(...DATA.sources.map((s) => s.ids));
  const protocols = [
    { cls: "", t: "Local 5-fold cross-validation", d: "70 official people in 5 identity-disjoint folds (14 held out each); fixed trial files for both tracks; 3 seeds × 5 folds at a fixed 90 epochs; paired identity-level bootstrap with 95% intervals." },
    { cls: "holdout", t: "Urdu holdout (development)", d: "Per fold, a fifth of MAV-Celeb v1 people are held out and no Urdu is used in training: unseen people in an unseen language. Used for model selection." },
    { cls: "holdout", t: "German holdout (confirmation)", d: "Same design with MAV-Celeb v3 and German. Candidates were frozen before it was run; it was never used for selection." },
    { cls: "official", t: "Official CodaBench", d: "English (heard) and Bangla (unheard), four cells. Used only to check finished systems, never for tuning; such scores are labelled “official”." },
  ];
  return (
    <Section id="data" n="02" eyebrow="DATA AND EVALUATION" band title="Training data and how every claim was tested"
      intro="External data is only ever added to training. Validation always uses held-out official identities, and the unseen-language tests hold out both the people and their language.">
      <div className="split">
        <div className="card">
          <p className="kicker">TRAINING IDENTITIES BY SOURCE</p>
          <ul className="bars">
            {DATA.sources.map((s) => (
              <li key={s.name}>
                <div className="row"><span>{s.name}</span><b className="num">{s.ids}</b></div>
                <div className="track"><div className={`fill ${s.ids === 70 ? "official" : ""}`} style={{ width: `${(100 * s.ids) / max}%` }} /></div>
                <small>{s.langs}{s.note ? ` · ${s.note}` : ""}</small>
              </li>
            ))}
          </ul>
          <p className="note" style={{ marginTop: 18 }}>
            {DATA.totalIds} identities, {DATA.totalRows.toLocaleString("en-US")} pairs (6,485 official + {DATA.externalRows.toLocaleString("en-US")} external). External rows by language: {DATA.externalByLanguage.map((l) => `${l.lang} ${l.rows.toLocaleString("en-US")}`).join(" · ")}.
            Clips detected as Bengali were dropped (VoxLingua107 language ID) and duplicates of official people removed. Raw media are not redistributed.
          </p>
        </div>
        <div className="stack">
          {protocols.map((p) => (
            <div key={p.t} className={`protocol ${p.cls}`}><h3>{p.t}</h3><p>{p.d}</p></div>
          ))}
        </div>
      </div>
    </Section>
  );
}
