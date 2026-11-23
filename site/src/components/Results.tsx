import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { expName, findings } from "../data/content";
import { exp, fmt, metric, REPO_URL } from "../data/registry";
import { Section } from "./Layout";
import { useSelection } from "./Journey";
import { AXIS_SMALL, GRID, LEGEND, SERIES_COLORS, TOOLTIP } from "./charts";

export function Findings() {
  const { select } = useSelection();
  return (
    <Section id="findings" n="06" band eyebrow="KEY DISCOVERIES" title="What the experiments taught us"
      intro="Each finding is backed by paired, multi-seed experiments; the caveat says what the evidence does not show.">
      <div className="challenge-grid three">
        {findings.map((f) => (
          <article key={f.n} className="challenge-card" style={{ flexDirection: "column", gap: 12 }}>
            <span className="challenge-number">0{f.n}</span>
            <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
              <p className="challenge-label">FINDING {f.n}</p>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
              <p className="caveat"><strong style={{ color: "#7a5a12" }}>Caveat:</strong> {f.caveat}</p>
              <div className="chips" style={{ marginTop: "auto", paddingTop: 14 }}>
                {f.refs.map((r) => (
                  <a key={r} href="#experiments" onClick={() => select(r)} className="chip chip-neutral" style={{ textDecoration: "none", textTransform: "none", letterSpacing: 0 }}>
                    {(() => { const t = expName(exp(r)); return t.length > 30 ? `${t.slice(0, 28)}…` : t; })()}
                  </a>
                ))}
              </div>
            </div>
          </article>
        ))}
      </div>
    </Section>
  );
}

export function FinalResults() {
  const rows = [
    { name: "B0 (expanded data, full model)",
      std: metric("b0_full", "B0 = + eGeMAPS + traits + language", "standard"), gen: metric("b0_full", "B0 = + eGeMAPS + traits + language", "gender"),
      terms: metric("b0_terms", "B0 + terms", "mean"),
      ur: metric("urdu_holdout_phase2", "B0", "gender_Urdu"), urs: metric("urdu_holdout_phase2", "B0", "standard_Urdu"),
      de: metric("german_holdout", "B0", "gender_German"), des: metric("german_holdout", "B0", "standard_German"), status: "eval #4 zip · not submitted" },
    { name: "C1 (+ ECAPA, single model)",
      std: metric("g3_cat_farl", "C1 = B0 voice + ECAPA (concat) + FaRL", "standard"), gen: metric("g3_cat_farl", "C1 = B0 voice + ECAPA (concat) + FaRL", "gender"),
      terms: metric("c1_terms", "C1 + terms", "mean"),
      ur: metric("urdu_holdout_phase2", "C1", "gender_Urdu"), urs: metric("urdu_holdout_phase2", "C1", "standard_Urdu"),
      de: metric("german_holdout", "C1", "gender_German"), des: metric("german_holdout", "C1", "standard_German"), status: "eval #5 zip · not submitted" },
    { name: "C1 × 3 (final ensemble)",
      std: metric("seed_ensemble_official", "C1 x 3", "standard"), gen: metric("seed_ensemble_official", "C1 x 3", "gender"),
      terms: metric("final_c1x3_terms", "C1 x 3 (raw-mean ensemble) + terms", "mean"),
      ur: metric("seed_ensembles_holdouts", "C1 x 3", "gender_Urdu", "systems_urdu"), urs: metric("seed_ensembles_holdouts", "C1 x 3", "standard_Urdu", "systems_urdu"),
      de: metric("seed_ensembles_holdouts", "C1 x 3", "gender_German", "systems_german"), des: metric("seed_ensembles_holdouts", "C1 x 3", "standard_German", "systems_german"), status: "eval #6 zip · not submitted" },
  ];
  const chart = [
    { k: "Local mean + terms", ...Object.fromEntries(rows.map((r) => [r.name, r.terms])) },
    { k: "Urdu same-gender (unseen)", ...Object.fromEntries(rows.map((r) => [r.name, r.ur])) },
    { k: "German same-gender (unseen)", ...Object.fromEntries(rows.map((r) => [r.name, r.de])) },
  ];
  const official = exp("codabench_eval_submissions");
  const lb = official.context!.leaderboard_13_nov;
  const [b0, , fin] = rows;
  return (
    <Section id="results" n="07" eyebrow="FINAL RESULTS" title="B0 → C1 → C1 × 3"
      intro="Local and holdout numbers are our own identity-disjoint validation, never official scores. Lower EER is better.">
      <div className="stats" style={{ maxWidth: "none" }} aria-label="Final system versus B0">
        <div><b className="num">{fmt(fin.terms)}%</b><span>C1 × 3 local mean EER with terms (B0: {fmt(b0.terms)}%)</span></div>
        <div><b className="num">{fmt(fin.ur)}%</b><span>same-gender EER on unseen Urdu (B0: {fmt(b0.ur)}%)</span></div>
        <div><b className="num">{fmt(fin.de)}%</b><span>same-gender EER on unseen German, never used for selection (B0: {fmt(b0.de)}%)</span></div>
      </div>
      <div className="table-wrap raised">
        <table className="wide">
          <thead>
            <tr><th>System</th><th className="num">Local std</th><th className="num">Local gender</th><th className="num">Local mean + terms</th><th className="num">Urdu std / gender</th><th className="num">German std / gender</th><th>Status</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.name} className={i === 2 ? "highlight strong" : ""}>
                <td>{r.name}</td>
                <td className="num">{fmt(r.std)}</td><td className="num">{fmt(r.gen)}</td>
                <td className="num"><b>{fmt(r.terms)}</b></td>
                <td className="num">{fmt(r.urs)} / {fmt(r.ur)}</td><td className="num">{fmt(r.des)} / {fmt(r.de)}</td>
                <td className="small">{r.status}</td>
              </tr>
            ))}
          </tbody>
          <caption>EER (%), 3 seeds × 5 folds. Local = official English folds; Urdu and German = unseen people in a language never used for training (German was never used for selection).</caption>
        </table>
      </div>
      <div className="split wide-left">
        <figure className="detail-visual" style={{ background: "#fff" }}>
          <div className="chart" style={{ height: 280 }} aria-label="Bar chart of the table's key columns">
            <ResponsiveContainer>
              <BarChart data={chart} margin={{ top: 10, right: 14, left: -12, bottom: 0 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="k" tick={AXIS_SMALL} />
                <YAxis tick={AXIS_SMALL} unit="%" domain={[20, 45]} ticks={[20, 25, 30, 35, 40, 45]} allowDataOverflow />
                <Tooltip contentStyle={TOOLTIP} formatter={(v: number) => `${v.toFixed(2)}%`} />
                <Legend wrapperStyle={LEGEND} />
                {rows.map((r, i) => <Bar key={r.name} dataKey={r.name} fill={SERIES_COLORS[i]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <figcaption>Key columns of the table. The y-axis starts at 20% to make differences visible.</figcaption>
        </figure>
        <div className="card">
          <p className="kicker" style={{ color: "#b88316" }}>OFFICIAL CODABENCH · VERIFIED</p>
          <div className="table-wrap" style={{ marginTop: 0 }}>
            <table>
              <tbody>
                {Object.entries(official.systems!).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td className="num"><b>{fmt(v.mean)}%</b></td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 14 }}>Evaluation phase, mean of four cells. Leaderboard on 13 Nov: {String(lb.entries)} entries, median {String(lb.median)}%, top {String(lb.top)}%; eval #2 ranked {String(lb.rank_of_eval2)}. No submission of B0, C1 or C1 × 3 is recorded and no final rank is available.</p>
        </div>
      </div>
    </Section>
  );
}

export function Limitations() {
  const items = [
    ["Small, uneven identity pools", "70 official people (40 men / 30 women); held-out folds have 14 people, so cells move by several points by chance. Every claim uses 3 seeds × 5 folds and identity-level resampling."],
    ["A voice does not determine a face", "Only shared traits (sex, age, build, possibly morphology) transfer. Same-gender, similar-age impostors remain the main error source; even the final system ranks the true face first among ~7 same-gender candidates less than half the time."],
    ["Attribute cues are coarse", "Gender and age terms help the standard track most; they are estimates from frozen models, and using them as matching cues encodes demographic priors."],
    ["Proxy languages are not Bangla", "Urdu and German holdouts use MAV-Celeb people and few speakers (≈ 10–13 per fold). Gains there need not transfer to Bangla."],
    ["English results disagree", "On the official English folds C1 is neutral, while the proxies' English cells improve; the reason is unresolved."],
    ["Model-selection uncertainty", "Many configurations were screened; intervals do not correct for that search. The German holdout was kept untouched to reduce this bias."],
    ["Pseudo-labelling failure", "With the current cross-modal signal, clustering-based pseudo-labels cannot reach usable precision; this would change only with a much stronger teacher."],
    ["Gap to the leaderboard", "Our best official score (28.84) is well behind the leaderboard median (22.47 on 13 Nov). Encoder fine-tuning and large paired pretraining (VoxCeleb dev) were not tried."],
  ];
  return (
    <Section id="limitations" n="08" band eyebrow="LIMITATIONS AND FUTURE WORK" title="What this work does not show">
      <div className="stage-grid two">
        {items.map(([t, d], i) => (
          <article key={t}><span>Limitation {i + 1}</span><h3>{t}</h3><p>{d}</p></article>
        ))}
      </div>
    </Section>
  );
}

export function Reproducibility() {
  const steps = [
    ["Environment", "conda create -n flag2027 python=3.10 && pip install -r requirements.txt"],
    ["Data (obtain yourself)", "FLAG 2027 train set + MAV-Celeb v1/v3 + VoxCeleb2-test — see docs/reproducibility/data.md"],
    ["Frozen features", "bash scripts/extract_features.sh   # FaRL, WavLM, eGeMAPS, ECAPA, age/gender"],
    ["Five-fold validation", "SEEDS=\"1 2 3\" bash scripts/runs/phase3_run_all.sh o_c1"],
    ["Paired comparison", "python FLAG_2027/local_validation/compare_fold_runs.py --run-a <B0> --run-b <C1> --seeds 1 2 3 ..."],
    ["Final models + zip", "bash scripts/train_final_c1x3.sh && bash scripts/make_submission_c1x3.sh"],
    ["Smoke test (no data, CPU)", "python -m pytest tests/ -q"],
  ];
  const links = [
    { t: "Repository and README", h: REPO_URL },
    { t: "Technical report", h: `${REPO_URL}/blob/main/docs/publication/report.md` },
    { t: "Research story", h: `${REPO_URL}/blob/main/docs/publication/research_story.md` },
    { t: "Experiment registry (JSON)", h: `${REPO_URL}/blob/main/public_results/experiments.json` },
    { t: "Licensing and data terms", h: `${REPO_URL}/blob/main/docs/reproducibility/licensing.md` },
  ];
  return (
    <>
      <Section id="reproduce" n="09" eyebrow="REPRODUCIBILITY" title="Code, data access and commands"
        intro="The pipeline is reproducible from documented, permitted components. Raw faces, voices and trained checkpoints are not redistributed: the datasets and several pretrained components (audeering age/gender, InsightFace, openSMILE) have their own access or non-commercial terms.">
        <div className="split wide-left">
          <ol className="steps">
            {steps.map(([t, c], i) => (
              <li key={t}><h3><span>{String(i + 1).padStart(2, "0")}</span>{t}</h3><pre><code>{c}</code></pre></li>
            ))}
          </ol>
          <ul className="link-list">
            {links.map((l) => <li key={l.t}><a href={l.h} target="_blank" rel="noreferrer">{l.t}<span aria-hidden="true">↗</span></a></li>)}
          </ul>
        </div>
      </Section>
      <section className="band">
        <div className="container section final-cta">
          <h2>Explore the Code</h2>
          <p>The repository holds the full pipeline, the experiment registry behind every number on this page, and the commands above.</p>
          <div className="buttons" style={{ marginBottom: 0 }}>
            <a className="button primary" href={REPO_URL} target="_blank" rel="noreferrer">View code ↗</a>
          </div>
        </div>
      </section>
    </>
  );
}
