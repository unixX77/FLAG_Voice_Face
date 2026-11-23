import { Fragment, useState } from "react";
import { Ban, Database, Flag, Lock, Sparkles, Sprout, TrendingUp } from "lucide-react";
import { architectures, type PipeNode } from "../data/content";
import { Section } from "./Layout";
import { B0Diagram, FinalDiagram, V1Diagram, V2Diagram } from "./FinalDiagram";

const DIAGRAMS: Record<string, typeof FinalDiagram> = { c1x3: FinalDiagram, v1: V1Diagram, v2: V2Diagram, b0: B0Diagram };

const LANES: Array<{ id: PipeNode["lane"]; label: string }> = [
  { id: "face", label: "Face branch" },
  { id: "voice", label: "Voice branch" },
  { id: "joint", label: "Scoring" },
  { id: "train", label: "Training only" },
];
const ICONS: Record<string, typeof Sprout> = { v1: Sprout, v2: TrendingUp, b0: Database, gamp: Ban, c1x3: Flag };
const STATUS_WORD = { final: "FINAL SYSTEM", rejected: "REJECTED", superseded: "SUPERSEDED", adopted: "ADOPTED" } as const;

function NodeButton({ n, active, onPin, onPreview }: { n: PipeNode; active: boolean; onPin: () => void; onPreview: (on: boolean) => void }) {
  return (
    <button type="button" className={`node k-${n.kind} ${active ? "is-active" : ""}`} aria-pressed={active}
      onClick={onPin} onMouseEnter={() => onPreview(true)} onMouseLeave={() => onPreview(false)} onFocus={() => onPreview(true)} onBlur={() => onPreview(false)}>
      <b>
        {n.kind === "frozen" && <Lock size={12} aria-label="frozen" />}
        {n.kind === "trainable" && <Sparkles size={12} aria-label="trainable" />}
        {n.label}
      </b>
      {n.sub && <small className="num">{n.sub}</small>}
    </button>
  );
}

/** Consecutive parallel nodes become one stacked group. */
function groupParallel(nodes: PipeNode[]): PipeNode[][] {
  const out: PipeNode[][] = [];
  for (const n of nodes) {
    const last = out[out.length - 1];
    if (n.parallel && last && last[0].parallel) last.push(n); else out.push([n]);
  }
  return out;
}

export function ArchitectureExplorer() {
  const [archId, setArchId] = useState("c1x3");
  const arch = architectures.find((a) => a.id === archId)!;
  const [pinned, setPinned] = useState<string | null>("concat");
  const [preview, setPreview] = useState<string | null>(null);
  const shown = arch.nodes.find((n) => n.id === (preview ?? pinned)) ?? null;
  const k = architectures.indexOf(arch);
  const statusCls = arch.status === "final" ? "final" : arch.status === "rejected" ? "rejected" : "";
  const choose = (id: string) => {
    const a = architectures.find((x) => x.id === id)!;
    setArchId(id); setPreview(null);
    setPinned(a.nodes.find((n) => n.kind === "trainable" || n.kind === "features")?.id ?? a.nodes[0].id);
  };

  return (
    <Section id="architecture" n="03" eyebrow="ARCHITECTURE EVOLUTION" title="How the system evolved, component by component"
      intro="Five versions, from the first baseline to the final ensemble, including one direction that was investigated and rejected. Dimensions of the final system were verified against the saved checkpoints and the inference code.">
      <div className="explorer">
        <div className="explorer-heading">
          <div>
            <p className="kicker">EXPLORE THE ARCHITECTURE</p>
            <h3>Five versions, one shared design</h3>
            <p>Select a version, then hover or tap a component to see what it does, whether it is trained, and the experiment behind it.</p>
          </div>
          <span className="explorer-progress">0{k + 1} — 0{architectures.length}</span>
        </div>
        <ol className="explorer-track arrows" role="tablist" aria-label="Architecture version">
          {architectures.map((a, i) => {
            const Icon = ICONS[a.id] ?? Sprout;
            const on = a.id === archId;
            return (
              <li key={a.id}>
                <button type="button" role="tab" aria-selected={on} className={`explorer-card ${on ? "is-active" : ""}`} onClick={() => choose(a.id)}>
                  <span className="card-top"><span className="card-number">0{i + 1}</span><Icon aria-hidden="true" /></span>
                  <span className={`card-tag ${a.status === "final" ? "final" : a.status === "rejected" ? "rejected" : ""}`}>{STATUS_WORD[a.status]}</span>
                  <strong>{a.name}</strong>
                  <span className="card-sub">{a.tag}</span>
                  <span className="card-action">{on ? "Selected version" : "Explore version"}<span aria-hidden="true">{on ? "✓" : "↗"}</span></span>
                </button>
              </li>
            );
          })}
        </ol>

        <div className="explorer-panel" role="tabpanel" aria-labelledby="arch-title">
          <div className="panel-header">
            <div>
              <span className={`detail-number ${statusCls}`}>VERSION 0{k + 1} · {STATUS_WORD[arch.status]}</span>
              <h4 id="arch-title">{arch.id === "c1x3" ? "Overview of the Final System (C1 × 3)" : `${arch.name} — ${arch.tag}`}</h4>
              <p>{arch.summary}</p>
            </div>
          </div>
          <div key={arch.id} className="panel-content single is-changing">
            <div>
              {arch.status === "rejected" && <p className="callout gold small" style={{ marginTop: 0 }}><strong>Investigated and rejected.</strong> Not part of the final model; nothing here is used at inference.</p>}
              {DIAGRAMS[arch.id] ? (
                (() => { const D = DIAGRAMS[arch.id]; return <D active={pinned} onPin={setPinned} onPreview={setPreview} />; })()
              ) : (
                <>
              <div className="lanes">
                {LANES.map((lane) => {
                  const nodes = arch.nodes.filter((n) => n.lane === lane.id);
                  if (!nodes.length) return null;
                  return (
                    <div key={lane.id}>
                      <p className="lane-label">{lane.label}</p>
                      <div className="lane-row">
                        {groupParallel(nodes).map((g, i) => (
                          <Fragment key={g.map((n) => n.id).join("+")}>
                            {i > 0 && lane.id !== "train" && <span className="lane-arrow" aria-hidden="true">→</span>}
                            {g.length === 1 ? (
                              <NodeButton n={g[0]} active={pinned === g[0].id} onPin={() => setPinned(g[0].id)} onPreview={(on) => setPreview(on ? g[0].id : null)} />
                            ) : (
                              <div className="parallel" aria-label="Run in parallel on the same clip">
                                {g.map((n) => <NodeButton key={n.id} n={n} active={pinned === n.id} onPin={() => setPinned(n.id)} onPreview={(on) => setPreview(on ? n.id : null)} />)}
                              </div>
                            )}
                          </Fragment>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
              <ul className="legend" aria-label="Legend">
                <li><i style={{ background: "#77a9dc" }} /> frozen pretrained encoder</li>
                <li><i style={{ background: "#1664c0" }} /> trained</li>
                <li><i style={{ background: "#9aa7b8" }} /> fixed features</li>
                <li><i style={{ background: "#b88316" }} /> scoring</li>
                <li><i style={{ background: "#8c6bb1" }} /> training-time only (dashed)</li>
              </ul>
                </>
              )}
              {arch.note && <p className="callout small">{arch.note}</p>}

              <div className="detail-copy" aria-live="polite" style={{ marginTop: 26, paddingTop: 22, borderTop: "1px solid #e3ebf4" }}>
                {shown ? (
                  <>
                    <span className="detail-number">COMPONENT · {shown.status}</span>
                    <h5 style={{ marginTop: 8, fontSize: "1.15rem" }}>{shown.label}</h5>
                    <p>{shown.what}</p>
                    <dl>
                      {shown.model && <div><dt>Model</dt><dd className="num">{shown.model}</dd></div>}
                      {shown.dims && <div><dt>Dimensions</dt><dd className="num">{shown.dims}</dd></div>}
                      {shown.why && <div><dt>Why</dt><dd>{shown.why}</dd></div>}
                      {shown.evidence && <div><dt>Evidence</dt><dd className="result">{shown.evidence}</dd></div>}
                      {shown.rejected && <div><dt>Alternatives rejected</dt><dd className="bad">{shown.rejected}</dd></div>}
                    </dl>
                  </>
                ) : (
                  <p>Hover or tap a component to see its details.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
