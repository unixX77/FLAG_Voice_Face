// Server-side render check (no browser needed): renders the whole page and every experiment's
// detail panel; any missing registry value throws. Also checks key numbers against the registry.
import { renderToString } from "react-dom/server";
import App from "./App";
import { Detail } from "./components/Explorer";
import { architectures, journey, findings } from "./data/content";
import { B0Diagram, FinalDiagram, V1Diagram, V2Diagram } from "./components/FinalDiagram";
import { experiments, metric, registry } from "./data/registry";

const html = renderToString(<App />).replace(/<!-- -->/g, "");  // React separates text nodes with empty comments
let fails = 0;
const must = (cond: boolean, msg: string) => { if (!cond) { console.error("FAIL:", msg); fails++; } };
for (const e of experiments) {
  try { renderToString(<Detail e={e} />); } catch (err) { console.error("FAIL detail", e.id, err); fails++; }
}
const final = metric("final_c1x3_terms", "C1 x 3 (raw-mean ensemble) + terms", "mean").toFixed(2);
must(html.includes(`${final}%`), `hero shows final metric ${final}`);
must(html.includes(registry.score_histograms.tracks.gender.eer.toFixed(2)), "EER explorer shows registry EER");
for (const a of architectures) for (const n of a.nodes) must(!!n.what, `${a.id}.${n.id} has a description`);
for (const s of journey) for (const r of s.refs) must(experiments.some((e) => e.id === r), `journey ${s.id} ref ${r} exists`);
for (const f of findings) for (const r of f.refs) must(experiments.some((e) => e.id === r), `finding ${f.n} ref ${r} exists`);
{
  // Every clickable module of the final-system diagram must open a real C1 × 3 component.
  const fd = renderToString(<FinalDiagram active={null} onPin={() => {}} onPreview={() => {}} />);
  const c1 = architectures.find((a) => a.id === "c1x3")!;
  const n = (fd.match(/class="fd-mod/g) ?? []).length;
  must(n === c1.nodes.length, `final diagram has ${n} modules for ${c1.nodes.length} C1 × 3 components`);
  must(!/768-d<\/em><\/button><span class="fd-arrow" aria-hidden="true">→<\/span><button[^>]*FaRL/.test(fd) && fd.includes("512-d") && fd.includes("128-d") && !/BatchNorm|BN|d = 512/.test(fd), "final diagram shows verified dimensions");
}
const v1 = renderToString(<V1Diagram active={null} onPin={() => {}} onPreview={() => {}} />);
{
  // V1 diagram: one module per V1 component and the verified V1 dimensions.
  const a = architectures.find((x) => x.id === "v1")!;
  const n = (v1.match(/class="fd-mod/g) ?? []).length;
  must(n === a.nodes.length, `V1 diagram has ${n} modules for ${a.nodes.length} V1 components`);
  must(v1.includes("Linear 1024 + LN + GELU") && v1.includes("256-d") && !/BatchNorm|FC \+ ArcFace|d = 512/.test(v1), "V1 diagram shows verified layers");
}
{
  // V2 diagram: one module per V2 component, verified projector (z-score → 256 → 128, L2) and no ArcFace.
  const v2 = renderToString(<V2Diagram active={null} onPin={() => {}} onPreview={() => {}} />);
  const a = architectures.find((x) => x.id === "v2")!;
  const n = (v2.match(/class="fd-mod/g) ?? []).length;
  must(n === a.nodes.length, `V2 diagram has ${n} modules for ${a.nodes.length} V2 components`);
  must(v2.includes("z-score") && v2.includes("768 → 128-d") && v2.includes("512 → 128-d") && /ArcFace identity loss.{0,400}removed in V2/.test(v2), "V2 diagram shows verified layers and ArcFace as removed");
  must((v2.match(/CHANGED IN V2/g) ?? []).length === 5 && v2.includes("What changed") && !v1.includes("CHANGED IN"), "V2 diagram marks the five changed modules; V1 has no change marks");
}
{
  // B0 diagram: one module per B0 component; verified 856-d voice input, refit B0 term weights, five change marks vs V2.
  const b0 = renderToString(<B0Diagram active={null} onPin={() => {}} onPreview={() => {}} />);
  const a = architectures.find((x) => x.id === "b0")!;
  const n = (b0.match(/class="fd-mod/g) ?? []).length;
  must(n === a.nodes.length, `B0 diagram has ${n} modules for ${a.nodes.length} B0 components`);
  must(b0.includes("856 → 128-d") && b0.includes("512 → 128-d") && b0.includes("301 identities") && !/ECAPA \(frozen\)|ArcFace/.test(b0), "B0 diagram shows verified layers");
  must(/gender track <!-- -->0.03<!-- --> \/ <!-- -->0</.test(b0) || b0.includes("gender track 0.03 / 0"), "B0 diagram uses B0's refit term weights");
  must((b0.match(/CHANGED IN B0|>NEW</g) ?? []).length === 5 && b0.includes("from <!-- -->V2"), "B0 diagram marks the five changed modules vs V2");
}
must(!/NaN|undefined|Infinity/.test(html), "no NaN/undefined/Infinity in rendered page");
must(!/COMPARE TWO SYSTEMS|Any two systems/.test(html), "system comparator removed");
must(!/C1 × 3[^<]{0,80}submitted to CodaBench/.test(html), "no claim that C1 × 3 was submitted");
console.log(`rendered ${html.length} chars; ${experiments.length} experiment panels; ${journey.length} journey steps; ${architectures.length} architectures; failures: ${fails}`);
process.exit(fails ? 1 : 0);
