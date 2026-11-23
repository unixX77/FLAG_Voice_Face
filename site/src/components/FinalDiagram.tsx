// Interactive overview of the final system (C1 × 3), drawn as a scalable SVG in the style of the project figure.
// Every module is a focusable button: hover previews and click pins its details (shown below by Architecture.tsx).
// Dimensions follow the verified checkpoints / inference code (docs/publication/01_audit.md):
// FaRL 512-d; voice = WavLM 768 | eGeMAPS 88 | ECAPA 192 = 1,048; projector z-score → Linear 256 → LayerNorm → GELU → Linear 128 → L2;
// per-seed distance 1 − cos, mean over 3 seeds, then fixed age/gender terms.
import type { KeyboardEvent, ReactNode } from "react";
import { exp } from "../data/registry";

interface Hit { active: string | null; onPin: (id: string) => void; onPreview: (id: string | null) => void }

const C = {
  voiceBg: "#fdf1f1", voiceLine: "#f0b9b9", faceBg: "#f1f6fd", faceLine: "#b7cfee",
  frozenBg: "#fff4d6", frozenLine: "#e7c063", featBg: "#dcebfa", featLine: "#94bbe3",
  projBg: "#ebe4fb", projLine: "#b19ce6", matchBg: "#f7f1fd", matchLine: "#d1bdf2",
  ensBg: "#eef5fd", ensLine: "#b5d1ef", termsBg: "#f2faee", termsLine: "#bcdfaa", outBg: "#fffaea", outLine: "#eed284",
  ink: "#1f2a37", box: "#c4cad3",
};

function Mod({ id, label, h, children }: { id: string; label: string; h: Hit; children: ReactNode }) {
  const on = h.active === id;
  const key = (e: KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); h.onPin(id); } };
  return (
    <g className={`fd-mod ${on ? "is-active" : ""}`} role="button" tabIndex={0} aria-label={label} aria-pressed={on}
      onClick={() => h.onPin(id)} onKeyDown={key} onMouseEnter={() => h.onPreview(id)} onMouseLeave={() => h.onPreview(null)}
      onFocus={() => h.onPreview(id)} onBlur={() => h.onPreview(null)}>
      {children}
    </g>
  );
}

/** Centre-anchored text. */
const T = ({ x, y, s = 20, w = 400, i = false, children }: { x: number; y: number; s?: number; w?: number; i?: boolean; children: ReactNode }) => (
  <text x={x} y={y} fontSize={s} fontWeight={w} fontStyle={i ? "italic" : "normal"} textAnchor="middle" dominantBaseline="middle">{children}</text>
);
/** Subscript via dy (baseline-shift is not supported everywhere). */
const Sub = ({ children }: { children: ReactNode }) => <><tspan dy={6} fontSize="0.72em">{children}</tspan><tspan dy={-6}>{"​"}</tspan></>;

const Box = ({ x, y, w, h, fill = "#fff", stroke = C.box, r = 12, dash = false, sw = 1.6 }: { x: number; y: number; w: number; h: number; fill?: string; stroke?: string; r?: number; dash?: boolean; sw?: number }) => (
  <rect className={dash ? undefined : "hit"} x={x} y={y} width={w} height={h} rx={r} fill={fill} stroke={stroke} strokeWidth={sw} strokeDasharray={dash ? "8 6" : undefined} />
);
const Chip = ({ x, y, w, h = 34, fill, stroke, children, s = 17 }: { x: number; y: number; w: number; h?: number; fill: string; stroke: string; children: ReactNode; s?: number }) => (
  <g><rect x={x} y={y} width={w} height={h} rx={7} fill={fill} stroke={stroke} strokeWidth={1.4} /><T x={x + w / 2} y={y + h / 2 + 1} s={s}>{children}</T></g>
);
const Arrow = ({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#111" strokeWidth={2.4} markerEnd="url(#fd-arrow)" />
);
const Stack = ({ x, y, w, n = 2 }: { x: number; y: number; w: number; n?: number }) => (
  <g>{Array.from({ length: n }, (_, k) => <rect key={k} x={x} y={y + k * 20} width={w} height={15} rx={2} fill="#c9def4" stroke="#86afd9" strokeWidth={1.3} />)}</g>
);
const Vec = ({ x, y }: { x: number; y: number }) => (
  <g>{Array.from({ length: 5 }, (_, k) => <rect key={k} x={x + k * 22} y={y} width={22} height={22} fill="#d8ccf6" stroke="#7656c4" strokeWidth={1.5} />)}</g>
);

function Wave({ cx, cy }: { cx: number; cy: number }) {
  const hs = [4, 7, 5, 11, 15, 9, 18, 12, 22, 14, 9, 17, 24, 13, 20, 11, 16, 8, 13, 19, 10, 15, 7, 11, 5, 8, 4];
  const x0 = cx - (hs.length * 4.6) / 2;
  return <g fill="#2f6fb0">{hs.map((v, k) => <rect key={k} x={x0 + k * 4.6} y={cy - v} width={2.6} height={2 * v} rx={1.2} />)}</g>;
}
function Avatar({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <rect x={x} y={y} width={90} height={62} rx={4} fill="#c9dcf1" stroke="#9fbcdc" strokeWidth={1.4} />
      <circle cx={x + 45} cy={y + 22} r={12} fill="#3f78b5" />
      <path d={`M${x + 20} ${y + 61}c3-14 12-21 25-21s22 7 25 21z`} fill="#3f78b5" />
    </g>
  );
}

export function FinalDiagram(h: Hit) {
  return (
    <div className="fd-scroll">
      <svg className="fd" viewBox="0 0 2172 790" role="group" aria-label="Overview of the final system (C1 × 3), interactive diagram"
        fontFamily="'Times New Roman', Times, 'Liberation Serif', 'Noto Serif', serif" fill={C.ink}>
        <defs>
          <marker id="fd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0L10 5L0 10z" fill="#111" />
          </marker>
        </defs>
        <rect x="6" y="6" width="2160" height="778" rx="22" fill="#fff" stroke="#dde3ea" strokeWidth={1.6} />

        {/* ---------------- voice branch ---------------- */}
        <T x={520} y={50} s={27} w={700}>Voice Branch (B<Sub>0</Sub> + ECAPA)</T>
        <Box x={36} y={74} w={962} h={214} fill={C.voiceBg} stroke={C.voiceLine} r={22} dash />
        <Mod id="v_in" label="Input voice clip" h={h}>
          <Box x={64} y={92} w={172} h={176} />
          <T x={150} y={116} s={21}>Input Voice Clip</T>
          <Wave cx={150} cy={176} />
          <T x={150} y={240} s={21}><tspan fontStyle="italic">x<Sub>v</Sub></tspan> (audio)</T>
        </Mod>
        <Arrow x1={236} y1={180} x2={282} y2={180} />
        <rect x={284} y={92} width={218} height={176} rx={12} fill="#fff" stroke={C.box} strokeWidth={1.6} />
        <T x={393} y={116} s={21} w={700}>B<Sub>0</Sub> Voice Features</T>
        <Mod id="wavlm" label="WavLM Base+, all-layer mean, 768-d" h={h}>
          <rect className="hit" x={296} y={142} width={194} height={46} rx={8} fill={C.featBg} stroke={C.featLine} strokeWidth={1.5} />
          <T x={393} y={166} s={18}>WavLM Base+ · 768</T>
        </Mod>
        <Mod id="egemaps" label="eGeMAPS v02, 88 functionals" h={h}>
          <rect className="hit" x={296} y={202} width={194} height={46} rx={8} fill={C.featBg} stroke={C.featLine} strokeWidth={1.5} />
          <T x={393} y={226} s={18}>eGeMAPS v02 · 88</T>
        </Mod>
        <Mod id="ecapa" label="ECAPA-TDNN, frozen, 192-d" h={h}>
          <Box x={516} y={92} w={132} h={176} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={582} y={118} s={21} w={700}>ECAPA</T>
          <T x={582} y={144} s={19}>(frozen)</T>
          <Stack x={541} y={170} w={82} />
          <T x={582} y={238} s={20}>192-d</T>
        </Mod>
        <Arrow x1={648} y1={180} x2={684} y2={180} />
        <Mod id="concat" label="Concatenation of 768, 88 and 192 to 1,048-d" h={h}>
          <circle className="hit" cx={708} cy={180} r={22} fill={C.projBg} stroke="#8f77d6" strokeWidth={1.8} />
          <T x={708} y={182} s={22} w={700}>C</T>
        </Mod>
        <Arrow x1={730} y1={180} x2={760} y2={180} />
        <Mod id="v_proj" label="Voice projection, one per seed" h={h}>
          <Box x={762} y={86} w={208} h={190} />
          <T x={866} y={108} s={21} w={700}>Voice Projection ×3</T>
          <Chip x={774} y={124} w={184} fill={C.projBg} stroke={C.projLine}>z-score</Chip>
          <Chip x={774} y={164} w={184} fill={C.projBg} stroke={C.projLine} s={16}>Linear 256 + LN + GELU</Chip>
          <Chip x={774} y={204} w={184} fill={C.projBg} stroke={C.projLine}>Linear 128 + L2</Chip>
          <T x={866} y={258} s={19}>1,048 → 128-d</T>
        </Mod>

        {/* ---------------- face branch ---------------- */}
        <T x={520} y={318} s={27} w={700}>Face Branch (FaRL)</T>
        <Box x={36} y={342} w={962} h={210} fill={C.faceBg} stroke={C.faceLine} r={22} dash />
        <Mod id="f_in" label="Input face image" h={h}>
          <Box x={84} y={360} w={198} h={174} />
          <T x={183} y={386} s={21} w={700}>Input Face Image</T>
          <Avatar x={138} y={410} />
          <T x={183} y={502} s={21}><tspan fontStyle="italic">x<Sub>f</Sub></tspan> (face)</T>
        </Mod>
        <Arrow x1={282} y1={447} x2={368} y2={447} />
        <Mod id="farl" label="FaRL ViT-B/16, frozen, 512-d" h={h}>
          <Box x={370} y={360} w={222} h={174} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={481} y={386} s={21} w={700}>FaRL (ViT-B/16)</T>
          <T x={481} y={412} s={19}>(frozen)</T>
          <Stack x={430} y={432} w={102} />
          <line x1={481} y1={470} x2={481} y2={490} stroke="#444" strokeWidth={1.5} strokeDasharray="3 3" />
          <T x={481} y={512} s={20}>512-d</T>
        </Mod>
        <Arrow x1={592} y1={447} x2={704} y2={447} />
        <Mod id="f_proj" label="Face projection, one per seed" h={h}>
          <Box x={706} y={352} w={240} h={190} />
          <T x={826} y={374} s={21} w={700}>Face Projection ×3</T>
          <Chip x={720} y={390} w={212} fill={C.projBg} stroke={C.projLine}>z-score</Chip>
          <Chip x={720} y={430} w={212} fill={C.projBg} stroke={C.projLine}>Linear 256 + LN + GELU</Chip>
          <Chip x={720} y={470} w={212} fill={C.projBg} stroke={C.projLine}>Linear 128 + L2</Chip>
          <T x={826} y={524} s={19}>512 → 128-d</T>
        </Mod>

        {/* joins into matching */}
        <polyline points="970,180 1012,180 1012,447 946,447" fill="none" stroke="#111" strokeWidth={2.4} />
        <Arrow x1={1012} y1={313} x2={1040} y2={313} />

        {/* ---------------- cross-modal matching ---------------- */}
        <Mod id="match" label="Cross-modal matching per seed: cosine distance" h={h}>
          <rect className="hit" x={1042} y={110} width={320} height={420} rx={22} fill={C.matchBg} stroke={C.matchLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1202} y={146} s={25} w={700}>Cross-Modal Matching</T>
          <text x={1066} y={244} fontSize={22} fontStyle="italic">z<Sub>v</Sub></text>
          <Vec x={1092} y={228} />
          <Arrow x1={1147} y1={258} x2={1147} y2={306} />
          <Arrow x1={1147} y1={306} x2={1147} y2={256} />
          <text x={1066} y={330} fontSize={22} fontStyle="italic">z<Sub>f</Sub></text>
          <Vec x={1092} y={314} />
          <rect x={1100} y={370} width={204} height={50} rx={12} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} />
          <T x={1202} y={396} s={21}>Cosine distance</T>
          <T x={1202} y={460} s={22}><tspan fontStyle="italic">d<Sub>k</Sub></tspan> = 1 − <tspan fontStyle="italic">z<Sub>v</Sub></tspan> · <tspan fontStyle="italic">z<Sub>f</Sub></tspan></T>
          <T x={1202} y={498} s={18}>per seed model <tspan fontStyle="italic">k</tspan></T>
        </Mod>
        <Arrow x1={1362} y1={313} x2={1388} y2={313} />

        {/* ---------------- 3-seed ensemble ---------------- */}
        <Mod id="cos" label="Three-seed ensemble: mean of the three distances" h={h}>
          <rect className="hit" x={1390} y={110} width={252} height={420} rx={22} fill={C.ensBg} stroke={C.ensLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1516} y={146} s={25} w={700}>3-Seed Ensemble</T>
          {[1, 2, 3].map((k) => (
            <g key={k}>
              <rect x={1406} y={196 + (k - 1) * 62} width={132} height={40} rx={9} fill="#d4e6f8" stroke="#8fb2d8" strokeWidth={1.4} />
              <T x={1472} y={217 + (k - 1) * 62} s={19}><tspan fontStyle="italic">d<Sub>{k}</Sub></tspan> (seed {k})</T>
            </g>
          ))}
          <path d="M1542 216 q16 0 16 16 v52 q0 16 12 16 q-12 0 -12 16 v52 q0 16 -16 16" fill="none" stroke="#111" strokeWidth={2.2} />
          <circle cx={1604} cy={300} r={26} fill={C.projBg} stroke="#8f77d6" strokeWidth={1.8} />
          <T x={1604} y={302} s={20} w={700}>1/3</T>
          <T x={1516} y={432} s={21}><tspan fontStyle="italic">d̄</tspan> = (<tspan fontStyle="italic">d</tspan><Sub>1</Sub> + <tspan fontStyle="italic">d</tspan><Sub>2</Sub> + <tspan fontStyle="italic">d</tspan><Sub>3</Sub>) / 3</T>
          <T x={1516} y={472} s={18}>equal weights</T>
        </Mod>
        <Arrow x1={1642} y1={313} x2={1666} y2={313} />

        {/* ---------------- age / gender terms ---------------- */}
        <Mod id="terms" label="Fixed age and gender terms added to the distance" h={h}>
          <rect className="hit" x={1668} y={110} width={276} height={420} rx={22} fill={C.termsBg} stroke={C.termsLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1806} y={138} s={25} w={700}>Age / Gender Terms</T>
          <T x={1806} y={168} s={19}>(fixed weights per track)</T>
          <rect x={1684} y={196} width={244} height={58} rx={10} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.6} />
          <T x={1806} y={226} s={23} w={700}><tspan fontStyle="italic">d</tspan> = <tspan fontStyle="italic">d̄</tspan> + <tspan fontStyle="italic">w<Sub>g</Sub></tspan>·<tspan fontStyle="italic">g</tspan> + <tspan fontStyle="italic">w<Sub>a</Sub></tspan>·<tspan fontStyle="italic">a</tspan></T>
          <rect x={1684} y={276} width={244} height={96} rx={10} fill="#fff" />
          <rect x={1694} y={300} width={48} height={48} rx={8} fill="#e6eef9" />
          <T x={1718} y={326} s={30}>♂</T>
          <text x={1752} y={312} fontSize={18}><tspan fontStyle="italic">g</tspan>: gender term</text>
          <text x={1752} y={336} fontSize={15}>|p<Sub>male</Sub>(voice) − male(face)|</text>
          <rect x={1684} y={388} width={244} height={96} rx={10} fill="#fff" />
          <rect x={1694} y={412} width={48} height={48} rx={8} fill="#e6eef9" />
          <g fill="#3f78b5"><circle cx={1711} cy={428} r={6} /><circle cx={1726} cy={428} r={6} /><path d="M1700 452c1-8 6-12 11-12s10 4 11 12zM1715 452c1-8 6-12 11-12s10 4 11 12z" /></g>
          <text x={1752} y={424} fontSize={18}><tspan fontStyle="italic">a</tspan>: age term</text>
          <text x={1752} y={448} fontSize={15}>|voice age − face age| / 10</text>
        </Mod>
        <Arrow x1={1944} y1={313} x2={1966} y2={313} />

        {/* ---------------- final output ---------------- */}
        <rect x={1968} y={110} width={186} height={420} rx={22} fill={C.outBg} stroke={C.outLine} strokeWidth={1.8} strokeDasharray="8 6" />
        <T x={2061} y={146} s={25} w={700}>Final Output</T>
        <rect x={1984} y={240} width={154} height={150} rx={14} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.6} />
        <T x={2061} y={276} s={34} w={700} i>d</T>
        <T x={2061} y={314} s={19}>(verification</T>
        <T x={2061} y={338} s={19}>distance; lower</T>
        <T x={2061} y={362} s={19}>= same person)</T>

        {/* ---------------- training only + legend ---------------- */}
        <Mod id="loss" label="Training-only losses, not used at inference" h={h}>
          <rect className="hit" x={1042} y={556} width={1112} height={54} rx={12} fill="#faf8fe" stroke={C.projLine} strokeWidth={1.6} strokeDasharray="8 6" />
          <T x={1598} y={584} s={20}><tspan fontWeight={700}>Training only:</tspan> SupCon (τ = 0.07) + pair + trait heads + language adversary · 301 identities · not used at inference</T>
        </Mod>
        <rect x={36} y={640} width={2118} height={58} rx={12} fill="#f5f6f8" stroke="#d6dae0" strokeWidth={1.4} />
        <g fontSize={20}>
          <text x={62} y={676} fontWeight={700} fontSize={22}>Legend:</text>
          <rect x={170} y={658} width={36} height={24} rx={5} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.5} /><text x={216} y={676}>Frozen pretrained encoder</text>
          <rect x={478} y={658} width={36} height={24} rx={5} fill={C.featBg} stroke={C.featLine} strokeWidth={1.5} /><text x={524} y={676}>Fixed features</text>
          <rect x={690} y={658} width={36} height={24} rx={5} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} /><text x={736} y={676}>Trained layers (×3 = one model per seed)</text>
          <circle cx={1150} cy={670} r={16} fill={C.projBg} stroke="#8f77d6" strokeWidth={1.6} /><text x={1150} y={677} textAnchor="middle" fontWeight={700} fontSize={18}>C</text>
          <text x={1176} y={676}>Concatenation (768 | 88 | 192 = 1,048)</text>
          <text x={1600} y={676}>LN: LayerNorm · 128-d: shared embedding</text>
        </g>
        <T x={1086} y={740} s={19}>Click any module for details. Clustering, pseudo-labels and score normalisation are not part of this system.</T>
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------
// Single-model versions (V1, V2), drawn with the same primitives, colours and layout as C1 × 3.
// V1 verified against runs/ablation_regularized/A_xvec_v1 (checkpoint config + weight shapes) and FLAG_2027/cross_modal_model.py:
//   x-vector 512 → [512, 512] → 256; FaRL 512 → [1024, 512] → 256; Linear + LayerNorm + GELU + Dropout(0.2, 0.1); L2 output;
//   ArcFace identity loss (56 fold identities, s 30, m 0.2) shared by both embeddings, training only.
// V2 verified against checkpoints/v2/full_data/checkpoint_full.pth.tar:
//   z-score → Linear 256 → LayerNorm → GELU → Dropout 0.3 → Linear 128 → L2 (face 512, voice 768 all-layer mean);
//   SupCon τ 0.07 + 0.25 pair; identity weight 0.
// ------------------------------------------------------------------------------------------------
const OutputPanel = ({ x, w = 186 }: { x: number; w?: number }) => (
  <g>
    <rect x={x} y={110} width={w} height={420} rx={22} fill={C.outBg} stroke={C.outLine} strokeWidth={1.8} strokeDasharray="8 6" />
    <T x={x + w / 2} y={146} s={25} w={700}>Final Output</T>
    <rect x={x + w / 2 - 77} y={240} width={154} height={150} rx={14} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.6} />
    <T x={x + w / 2} y={276} s={34} w={700} i>d</T>
    <T x={x + w / 2} y={314} s={19}>(verification</T>
    <T x={x + w / 2} y={338} s={19}>distance; lower</T>
    <T x={x + w / 2} y={362} s={19}>= same person)</T>
  </g>
);

type Row = [string, string, ("new" | "removed")?];
interface SingleSpec {
  aria: string;
  branchTitle: string;
  voiceEncoder: { id: string; label: string; title: string; dims: string };
  voiceProj: { chips: string[]; caption: string };
  faceProj: { chips: string[]; caption: string };
  embedding: string;
  train: { label: string; rows: Row[]; footer: [string, string]; legend: string };
  note: string;
  changes?: Changes;
}
/** Highlights relative to the previous version: badged module ids, gold chips, and a summary strip. */
interface Changes { tag: string; from: string; ids: string[]; newChips: string[]; items: Array<[string, string]> }

const GOLD = { bg: "#ffffff", line: "#d9a521", ink: "#7a5600" };
/** Module boxes in the single-model layout (for change badges). */
const SINGLE_BOX: Record<string, [number, number, number, number]> = {
  enc: [332, 92, 260, 176], v_proj: [702, 86, 244, 190], f_proj: [702, 352, 244, 190], cos: [1042, 110, 498, 420], loss: [1784, 110, 370, 420],
};
type Rect = [number, number, number, number];
function ChangeMark({ box: [x, y, w, h], tag: full }: { box: Rect; tag: string }) {
  const tag = w < 150 ? "NEW" : full;
  const bw = 16 + tag.length * 9.4;
  return (
    <g pointerEvents="none" fontFamily="'Noto Sans', 'Google Sans', Arial, sans-serif">
      <rect x={x - 5} y={y - 5} width={w + 10} height={h + 10} rx={16} fill="none" stroke={GOLD.line} strokeWidth={3} />
      <rect x={x + w - bw - 6} y={y - 19} width={bw} height={28} rx={14} fill={GOLD.line} />
      <text x={x + w - bw / 2 - 6} y={y - 4} fontSize={14} fontWeight={700} fill="#fff" textAnchor="middle" dominantBaseline="middle" letterSpacing="0.06em">{tag}</text>
    </g>
  );
}
function ChangeStrip({ ch, y }: { ch: Changes; y: number }) {
  const col = (2154 - 290) / ch.items.length;
  return (
    <g>
      <rect x={36} y={y} width={2118} height={100} rx={12} fill="#fffaea" stroke={GOLD.line} strokeWidth={1.8} />
      <text x={62} y={y + 40} fontSize={22} fontWeight={700} fill={GOLD.ink}>What changed</text>
      <text x={62} y={y + 68} fontSize={22} fontWeight={700} fill={GOLD.ink}>from {ch.from}:</text>
      {ch.items.map(([a, b], k) => (
        <g key={a}>
          <text x={290 + k * col} y={y + 38} fontSize={20} fontWeight={700}>{k + 1}. {a}</text>
          <text x={290 + k * col} y={y + 68} fontSize={18}>{b}</text>
        </g>
      ))}
    </g>
  );
}

function SingleModelDiagram({ spec, ...h }: Hit & { spec: SingleSpec }) {
  const ch = spec.changes;
  const isNew = (c: string) => !!ch?.newChips.includes(c);
  const chips = (x: number, y0: number, list: string[]) => list.map((c, k) => (
    <g key={k}>
      <Chip x={x} y={y0 + k * 40} w={216} fill={isNew(c) ? GOLD.bg : C.projBg} stroke={isNew(c) ? GOLD.line : C.projLine} s={c.length > 21 ? 16 : 17}>{c}</Chip>
      {isNew(c) && <rect x={x} y={y0 + k * 40} width={216} height={34} rx={7} fill="none" stroke={GOLD.line} strokeWidth={2.6} pointerEvents="none" />}
    </g>
  ));
  const rowH = spec.train.rows.length > 2 ? 86 : 110;
  const H = ch ? 800 : 700;
  const boxOf = (id: string) => SINGLE_BOX[id === spec.voiceEncoder.id ? "enc" : id];
  return (
    <div className="fd-scroll">
      <svg className="fd" viewBox={`0 0 2172 ${H}`} role="group" aria-label={spec.aria}
        fontFamily="'Times New Roman', Times, 'Liberation Serif', 'Noto Serif', serif" fill={C.ink}>
        <defs>
          <marker id="fd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0L10 5L0 10z" fill="#111" />
          </marker>
        </defs>
        <rect x="6" y="6" width="2160" height={H - 12} rx="22" fill="#fff" stroke="#dde3ea" strokeWidth={1.6} />

        {/* voice branch */}
        <T x={520} y={50} s={27} w={700}>{spec.branchTitle}</T>
        <Box x={36} y={74} w={962} h={214} fill={C.voiceBg} stroke={C.voiceLine} r={22} dash />
        <Mod id="v_in" label="Input voice clip" h={h}>
          <Box x={64} y={92} w={172} h={176} />
          <T x={150} y={116} s={21}>Input Voice Clip</T>
          <Wave cx={150} cy={176} />
          <T x={150} y={240} s={21}><tspan fontStyle="italic">x<Sub>v</Sub></tspan> (audio)</T>
        </Mod>
        <Arrow x1={236} y1={180} x2={330} y2={180} />
        <Mod id={spec.voiceEncoder.id} label={spec.voiceEncoder.label} h={h}>
          <Box x={332} y={92} w={260} h={176} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={462} y={118} s={21} w={700}>{spec.voiceEncoder.title}</T>
          <T x={462} y={144} s={19}>(frozen)</T>
          <Stack x={411} y={170} w={102} />
          <T x={462} y={238} s={20}>{spec.voiceEncoder.dims}</T>
        </Mod>
        <Arrow x1={592} y1={180} x2={700} y2={180} />
        <Mod id="v_proj" label="Voice projector" h={h}>
          <Box x={702} y={86} w={244} h={190} />
          <T x={824} y={108} s={21} w={700}>Voice Projector</T>
          {chips(716, 124, spec.voiceProj.chips)}
          <T x={824} y={258} s={19}>{spec.voiceProj.caption}</T>
        </Mod>

        {/* face branch */}
        <T x={520} y={318} s={27} w={700}>Face Branch (FaRL)</T>
        <Box x={36} y={342} w={962} h={210} fill={C.faceBg} stroke={C.faceLine} r={22} dash />
        <Mod id="f_in" label="Input face image" h={h}>
          <Box x={84} y={360} w={198} h={174} />
          <T x={183} y={386} s={21} w={700}>Input Face Image</T>
          <Avatar x={138} y={410} />
          <T x={183} y={502} s={21}><tspan fontStyle="italic">x<Sub>f</Sub></tspan> (face)</T>
        </Mod>
        <Arrow x1={282} y1={447} x2={368} y2={447} />
        <Mod id="farl" label="FaRL ViT-B/16, frozen, 512-d" h={h}>
          <Box x={370} y={360} w={222} h={174} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={481} y={386} s={21} w={700}>FaRL (ViT-B/16)</T>
          <T x={481} y={412} s={19}>(frozen)</T>
          <Stack x={430} y={432} w={102} />
          <line x1={481} y1={470} x2={481} y2={490} stroke="#444" strokeWidth={1.5} strokeDasharray="3 3" />
          <T x={481} y={512} s={20}>512-d</T>
        </Mod>
        <Arrow x1={592} y1={447} x2={700} y2={447} />
        <Mod id="f_proj" label="Face projector" h={h}>
          <Box x={702} y={352} w={244} h={190} />
          <T x={824} y={374} s={21} w={700}>Face Projector</T>
          {chips(716, 390, spec.faceProj.chips)}
          <T x={824} y={524} s={19}>{spec.faceProj.caption}</T>
        </Mod>

        {/* join */}
        <polyline points="946,180 1012,180 1012,447 946,447" fill="none" stroke="#111" strokeWidth={2.4} />
        <Arrow x1={1012} y1={313} x2={1040} y2={313} />

        {/* matching (single model) */}
        <Mod id="cos" label="Cross-modal matching: cosine distance, single model" h={h}>
          <rect className="hit" x={1042} y={110} width={498} height={420} rx={22} fill={C.matchBg} stroke={C.matchLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1291} y={146} s={25} w={700}>Cross-Modal Matching</T>
          <text x={1070} y={254} fontSize={22} fontStyle="italic">z<Sub>v</Sub></text>
          <Vec x={1096} y={238} />
          <Arrow x1={1151} y1={268} x2={1151} y2={306} />
          <Arrow x1={1151} y1={306} x2={1151} y2={266} />
          <text x={1070} y={340} fontSize={22} fontStyle="italic">z<Sub>f</Sub></text>
          <Vec x={1096} y={324} />
          <Arrow x1={1222} y1={292} x2={1292} y2={292} />
          <rect x={1296} y={262} width={214} height={60} rx={12} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} />
          <T x={1403} y={293} s={21}>Cosine distance</T>
          <T x={1291} y={410} s={23}><tspan fontStyle="italic">d</tspan> = 1 − <tspan fontStyle="italic">z<Sub>v</Sub></tspan> · <tspan fontStyle="italic">z<Sub>f</Sub></tspan></T>
          <T x={1291} y={452} s={18}>one model · {spec.embedding} unit vectors</T>
        </Mod>
        <Arrow x1={1540} y1={313} x2={1566} y2={313} />
        <OutputPanel x={1568} />

        {/* training only */}
        <Mod id="loss" label={spec.train.label} h={h}>
          <rect className="hit" x={1784} y={110} width={370} height={420} rx={22} fill="#faf8fe" stroke={C.projLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1969} y={146} s={25} w={700}>Training Only</T>
          {spec.train.rows.map(([a, b, st], k) => (
            <g key={a} opacity={st === "removed" ? 0.75 : 1}>
              <rect x={1800} y={180 + k * rowH} width={338} height={74} rx={10}
                fill={st === "new" ? GOLD.bg : st === "removed" ? "#f1f2f4" : "#fff"} stroke={st === "new" ? GOLD.line : st === "removed" ? "#b8bec7" : C.projLine}
                strokeWidth={st === "new" ? 2.6 : 1.3} strokeDasharray={st === "removed" ? "6 4" : undefined} />
              <text x={1969} y={206 + k * rowH} fontSize={19} fontWeight={700} textAnchor="middle" dominantBaseline="middle"
                textDecoration={st === "removed" ? "line-through" : undefined} fill={st === "removed" ? "#6b7380" : undefined}>{a}</text>
              <T x={1969} y={232 + k * rowH} s={16}>{b}</T>
            </g>
          ))}
          <T x={1969} y={470} s={18}>{spec.train.footer[0]}</T>
          <T x={1969} y={496} s={18}>{spec.train.footer[1]}</T>
        </Mod>

        {/* legend */}
        <rect x={36} y={566} width={2118} height={58} rx={12} fill="#f5f6f8" stroke="#d6dae0" strokeWidth={1.4} />
        <g fontSize={20}>
          <text x={62} y={602} fontWeight={700} fontSize={22}>Legend:</text>
          <rect x={170} y={584} width={36} height={24} rx={5} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.5} /><text x={216} y={602}>Frozen pretrained encoder</text>
          <rect x={478} y={584} width={36} height={24} rx={5} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} /><text x={524} y={602}>Trained layers (one model)</text>
          <rect x={800} y={584} width={36} height={24} rx={5} fill="#faf8fe" stroke={C.projLine} strokeWidth={1.5} strokeDasharray="5 4" /><text x={846} y={602}>{spec.train.legend}</text>
          {ch && <><rect x={1250} y={584} width={36} height={24} rx={5} fill={GOLD.bg} stroke={GOLD.line} strokeWidth={3} /><text x={1296} y={602}>Changed from {ch.from}</text></>}
          <text x={2130} y={602} textAnchor="end">LN: LayerNorm · {spec.embedding}: shared embedding</text>
        </g>
        {ch && <ChangeStrip ch={ch} y={640} />}
        {ch?.ids.map((id) => <ChangeMark key={id} box={boxOf(id)} tag={ch.tag} />)}
        <T x={1086} y={H - 36} s={19}>{spec.note}</T>
      </svg>
    </div>
  );
}

const V1_SPEC: SingleSpec = {
  aria: "V1 initial baseline, interactive diagram",
  branchTitle: "Voice Branch (WavLM-SV x-vector)",
  voiceEncoder: { id: "xvec", label: "WavLM-SV x-vector, frozen, 512-d", title: "WavLM-SV x-vector", dims: "512-d" },
  voiceProj: { chips: ["Linear 512 + LN + GELU", "Linear 512 + LN + GELU", "Linear 256 + L2"], caption: "512 → 256-d" },
  faceProj: { chips: ["Linear 1024 + LN + GELU", "Linear 512 + LN + GELU", "Linear 256 + L2"], caption: "512 → 256-d" },
  embedding: "256-d",
  train: {
    label: "Training-only losses: multi-positive contrastive, ArcFace identity and pair loss",
    rows: [["Multi-positive contrastive", "cross-modal, τ = 0.07 · weight 1.0"], ["ArcFace identity loss", "shared classifier · 56 identities · s 30, m 0.2"], ["Pair loss", "weight 0.25"]],
    footer: ["70 official identities · early stopping", "not used at inference"],
    legend: "Training only (ArcFace, contrastive and pair losses)",
  },
  note: "Compared with C1 × 3: no input standardisation, no eGeMAPS or ECAPA, no seed ensemble and no age/gender terms.",
};

const V2_SPEC: SingleSpec = {
  aria: "V2 improved representation, interactive diagram",
  branchTitle: "Voice Branch (WavLM Base+)",
  voiceEncoder: { id: "wavlm", label: "WavLM Base+, frozen, all-layer mean, 768-d", title: "WavLM Base+", dims: "all-layer mean · 768-d" },
  voiceProj: { chips: ["z-score", "Linear 256 + LN + GELU", "Linear 128 + L2"], caption: "768 → 128-d" },
  faceProj: { chips: ["z-score", "Linear 256 + LN + GELU", "Linear 128 + L2"], caption: "512 → 128-d" },
  embedding: "128-d",
  train: {
    label: "Training-only losses: supervised contrastive and pair loss",
    rows: [["SupCon (symmetric)", "τ = 0.07 · all same-identity samples positive", "new"], ["ArcFace identity loss", "removed in V2 (weight 0)", "removed"], ["Pair loss", "weight 0.25 (unchanged)"]],
    footer: ["70 official identities · 90 epochs", "not used at inference"],
    legend: "Training only (SupCon and pair losses)",
  },
  note: "Same projector design as C1 × 3; no eGeMAPS or ECAPA, no trait or language heads, no external data, no seed ensemble and no age/gender terms.",
  changes: {
    tag: "CHANGED IN V2", from: "V1", ids: ["wavlm", "v_proj", "f_proj", "cos", "loss"], newChips: ["z-score"],
    items: [
      ["Voice encoder", "x-vector 512-d → WavLM Base+ all-layer mean 768-d"],
      ["Projectors", "3 layers, 256-d → z-score + 2 layers, 128-d"],
      ["Training loss", "ArcFace removed · SupCon replaces contrastive"],
    ],
  },
};

export const V1Diagram = (h: Hit) => <SingleModelDiagram {...h} spec={V1_SPEC} />;
export const V2Diagram = (h: Hit) => <SingleModelDiagram {...h} spec={V2_SPEC} />;

// ------------------------------------------------------------------------------------------------
// B0 (eval #4 build), verified against checkpoints/v4_traits/full_data/checkpoint_full.pth.tar + README:
//   voice 856 = WavLM all-layer mean 768 | eGeMAPS v02 88; same projectors as V2 (z-score → 256 → 128, L2), face 512;
//   SupCon τ 0.07 + 0.25 pair + 0.2 trait heads (age/build/gender, face and voice) + 0.1 language adversary
//   + 0.1 cross-lingual alignment (voice); 301 identities, 50% official per batch, 50 batches × 90 epochs; one model;
//   fixed age/gender terms refit on runs/traits/full (checkpoints/v4_traits/attribute_terms.json).
// ------------------------------------------------------------------------------------------------
const B0_CHANGES: Changes = {
  tag: "CHANGED IN B0", from: "V2", ids: ["egemaps", "concat", "v_proj", "terms", "loss"], newChips: [],
  items: [
    ["Training data", "70 → 301 identities · 50% official per batch"],
    ["Voice input", "+ eGeMAPS (88) → 768 | 88 = 856-d"],
    ["Training loss", "+ trait heads · language adversary · cross-lingual"],
    ["Scoring", "+ fixed age / gender terms"],
  ],
};
const B0_BOX: Record<string, Rect> = {
  egemaps: [284, 188, 276, 80], concat: [618, 158, 44, 44], v_proj: [702, 86, 244, 190], terms: [1500, 110, 320, 420], loss: [1042, 548, 1112, 80],
};

export function B0Diagram(h: Hit) {
  const w = exp("b0_terms").term_weights!;
  const ch = B0_CHANGES;
  const H = 900;
  const projChips = (x: number, y0: number) => ["z-score", "Linear 256 + LN + GELU", "Linear 128 + L2"].map((c, k) => (
    <Chip key={k} x={x} y={y0 + k * 40} w={216} fill={C.projBg} stroke={C.projLine} s={c.length > 21 ? 16 : 17}>{c}</Chip>
  ));
  return (
    <div className="fd-scroll">
      <svg className="fd" viewBox={`0 0 2172 ${H}`} role="group" aria-label="B0 expanded data, full model, interactive diagram"
        fontFamily="'Times New Roman', Times, 'Liberation Serif', 'Noto Serif', serif" fill={C.ink}>
        <defs>
          <marker id="fd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0L10 5L0 10z" fill="#111" />
          </marker>
        </defs>
        <rect x="6" y="6" width="2160" height={H - 12} rx="22" fill="#fff" stroke="#dde3ea" strokeWidth={1.6} />

        {/* voice branch */}
        <T x={520} y={50} s={27} w={700}>Voice Branch (WavLM + eGeMAPS)</T>
        <Box x={36} y={74} w={962} h={214} fill={C.voiceBg} stroke={C.voiceLine} r={22} dash />
        <Mod id="v_in" label="Input voice clip" h={h}>
          <Box x={64} y={92} w={172} h={176} />
          <T x={150} y={116} s={21}>Input Voice Clip</T>
          <Wave cx={150} cy={176} />
          <T x={150} y={240} s={21}><tspan fontStyle="italic">x<Sub>v</Sub></tspan> (audio)</T>
        </Mod>
        <polyline points="236,180 258,180 258,132 258,228" fill="none" stroke="#111" strokeWidth={2.4} />
        <Arrow x1={258} y1={132} x2={282} y2={132} />
        <Arrow x1={258} y1={228} x2={282} y2={228} />
        <Mod id="wavlm" label="WavLM Base+, frozen, all-layer mean, 768-d" h={h}>
          <Box x={284} y={92} w={276} h={80} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={422} y={118} s={21} w={700}>WavLM Base+ (frozen)</T>
          <T x={422} y={150} s={19}>all-layer mean · 768-d</T>
        </Mod>
        <Mod id="egemaps" label="eGeMAPS v02, 88 fixed functionals" h={h}>
          <Box x={284} y={188} w={276} h={80} fill={C.featBg} stroke={C.featLine} />
          <T x={422} y={214} s={21} w={700}>eGeMAPS v02</T>
          <T x={422} y={246} s={19}>88 functionals (fixed)</T>
        </Mod>
        <polyline points="560,132 588,132 588,228 560,228" fill="none" stroke="#111" strokeWidth={2.4} />
        <Arrow x1={588} y1={180} x2={616} y2={180} />
        <Mod id="concat" label="Concatenation of 768 and 88 to 856-d" h={h}>
          <circle className="hit" cx={640} cy={180} r={22} fill={C.projBg} stroke="#8f77d6" strokeWidth={1.8} />
          <T x={640} y={182} s={22} w={700}>C</T>
          <T x={640} y={226} s={17}>856-d</T>
        </Mod>
        <Arrow x1={662} y1={180} x2={700} y2={180} />
        <Mod id="v_proj" label="Voice projector" h={h}>
          <Box x={702} y={86} w={244} h={190} />
          <T x={824} y={108} s={21} w={700}>Voice Projector</T>
          {projChips(716, 124)}
          <T x={824} y={258} s={19}>856 → 128-d</T>
        </Mod>

        {/* face branch */}
        <T x={520} y={318} s={27} w={700}>Face Branch (FaRL)</T>
        <Box x={36} y={342} w={962} h={210} fill={C.faceBg} stroke={C.faceLine} r={22} dash />
        <Mod id="f_in" label="Input face image" h={h}>
          <Box x={84} y={360} w={198} h={174} />
          <T x={183} y={386} s={21} w={700}>Input Face Image</T>
          <Avatar x={138} y={410} />
          <T x={183} y={502} s={21}><tspan fontStyle="italic">x<Sub>f</Sub></tspan> (face)</T>
        </Mod>
        <Arrow x1={282} y1={447} x2={368} y2={447} />
        <Mod id="farl" label="FaRL ViT-B/16, frozen, 512-d" h={h}>
          <Box x={370} y={360} w={222} h={174} fill={C.frozenBg} stroke={C.frozenLine} />
          <T x={481} y={386} s={21} w={700}>FaRL (ViT-B/16)</T>
          <T x={481} y={412} s={19}>(frozen)</T>
          <Stack x={430} y={432} w={102} />
          <line x1={481} y1={470} x2={481} y2={490} stroke="#444" strokeWidth={1.5} strokeDasharray="3 3" />
          <T x={481} y={512} s={20}>512-d</T>
        </Mod>
        <Arrow x1={592} y1={447} x2={700} y2={447} />
        <Mod id="f_proj" label="Face projector" h={h}>
          <Box x={702} y={352} w={244} h={190} />
          <T x={824} y={374} s={21} w={700}>Face Projector</T>
          {projChips(716, 390)}
          <T x={824} y={524} s={19}>512 → 128-d</T>
        </Mod>

        {/* join */}
        <polyline points="946,180 1012,180 1012,447 946,447" fill="none" stroke="#111" strokeWidth={2.4} />
        <Arrow x1={1012} y1={313} x2={1040} y2={313} />

        {/* matching (single model) */}
        <Mod id="cos" label="Cross-modal matching: cosine distance, single model" h={h}>
          <rect className="hit" x={1042} y={110} width={430} height={420} rx={22} fill={C.matchBg} stroke={C.matchLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1257} y={146} s={25} w={700}>Cross-Modal Matching</T>
          <text x={1066} y={254} fontSize={22} fontStyle="italic">z<Sub>v</Sub></text>
          <Vec x={1092} y={238} />
          <Arrow x1={1147} y1={268} x2={1147} y2={306} />
          <Arrow x1={1147} y1={306} x2={1147} y2={266} />
          <text x={1066} y={340} fontSize={22} fontStyle="italic">z<Sub>f</Sub></text>
          <Vec x={1092} y={324} />
          <Arrow x1={1210} y1={292} x2={1256} y2={292} />
          <rect x={1260} y={262} width={192} height={60} rx={12} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} />
          <T x={1356} y={293} s={20}>Cosine distance</T>
          <T x={1257} y={410} s={23}><tspan fontStyle="italic">d<Sub>cos</Sub></tspan> = 1 − <tspan fontStyle="italic">z<Sub>v</Sub></tspan> · <tspan fontStyle="italic">z<Sub>f</Sub></tspan></T>
          <T x={1257} y={452} s={18}>one model · 128-d unit vectors</T>
        </Mod>
        <Arrow x1={1472} y1={313} x2={1498} y2={313} />

        {/* age / gender terms */}
        <Mod id="terms" label="Fixed age and gender terms added to the distance" h={h}>
          <rect className="hit" x={1500} y={110} width={320} height={420} rx={22} fill={C.termsBg} stroke={C.termsLine} strokeWidth={1.8} strokeDasharray="8 6" />
          <T x={1660} y={138} s={25} w={700}>Age / Gender Terms</T>
          <T x={1660} y={168} s={19}>(fixed weights per track)</T>
          <rect x={1516} y={190} width={288} height={58} rx={10} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.6} />
          <T x={1660} y={220} s={22} w={700}><tspan fontStyle="italic">d</tspan> = <tspan fontStyle="italic">d<Sub>cos</Sub></tspan> + <tspan fontStyle="italic">w<Sub>g</Sub></tspan>·<tspan fontStyle="italic">g</tspan> + <tspan fontStyle="italic">w<Sub>a</Sub></tspan>·<tspan fontStyle="italic">a</tspan></T>
          <rect x={1516} y={262} width={288} height={84} rx={10} fill="#fff" />
          <rect x={1526} y={280} width={48} height={48} rx={8} fill="#e6eef9" />
          <T x={1550} y={306} s={30}>♂</T>
          <text x={1586} y={294} fontSize={18}><tspan fontStyle="italic">g</tspan>: gender term</text>
          <text x={1586} y={318} fontSize={15}>|p<Sub>male</Sub>(voice) − male(face)|</text>
          <rect x={1516} y={358} width={288} height={84} rx={10} fill="#fff" />
          <rect x={1526} y={376} width={48} height={48} rx={8} fill="#e6eef9" />
          <g fill="#3f78b5"><circle cx={1543} cy={392} r={6} /><circle cx={1558} cy={392} r={6} /><path d="M1532 416c1-8 6-12 11-12s10 4 11 12zM1547 416c1-8 6-12 11-12s10 4 11 12z" /></g>
          <text x={1586} y={390} fontSize={18}><tspan fontStyle="italic">a</tspan>: age term</text>
          <text x={1586} y={414} fontSize={15}>|voice age − face age| / 10</text>
          <T x={1660} y={472} s={16}><tspan fontStyle="italic">w<Sub>a</Sub></tspan> / <tspan fontStyle="italic">w<Sub>g</Sub></tspan>: standard {w.standard_track.w_age_per_10y} / {w.standard_track.w_gender}</T>
          <T x={1660} y={498} s={16}>gender track {w.gender_track.w_age_per_10y} / {w.gender_track.w_gender}</T>
        </Mod>
        <Arrow x1={1820} y1={313} x2={1846} y2={313} />
        <OutputPanel x={1848} w={306} />

        {/* training only */}
        <Mod id="loss" label="Training-only losses: SupCon, pair, trait heads and language losses" h={h}>
          <rect className="hit" x={1042} y={548} width={1112} height={80} rx={12} fill="#faf8fe" stroke={C.projLine} strokeWidth={1.6} strokeDasharray="8 6" />
          <T x={1598} y={574} s={20}><tspan fontWeight={700}>Training only</tspan> (not used at inference) · 301 identities · 50% official per batch · 90 epochs</T>
          <T x={1598} y={604} s={18}>SupCon (τ = 0.07) + 0.25 pair + 0.2 trait heads (age, build, gender) + 0.1 language adversary + 0.1 cross-lingual</T>
        </Mod>

        {/* legend */}
        <rect x={36} y={648} width={2118} height={58} rx={12} fill="#f5f6f8" stroke="#d6dae0" strokeWidth={1.4} />
        <g fontSize={20}>
          <text x={62} y={684} fontWeight={700} fontSize={22}>Legend:</text>
          <rect x={170} y={666} width={36} height={24} rx={5} fill={C.frozenBg} stroke={C.frozenLine} strokeWidth={1.5} /><text x={216} y={684}>Frozen pretrained encoder</text>
          <rect x={478} y={666} width={36} height={24} rx={5} fill={C.featBg} stroke={C.featLine} strokeWidth={1.5} /><text x={524} y={684}>Fixed features</text>
          <rect x={690} y={666} width={36} height={24} rx={5} fill={C.projBg} stroke={C.projLine} strokeWidth={1.5} /><text x={736} y={684}>Trained layers (one model)</text>
          <rect x={1000} y={666} width={36} height={24} rx={5} fill="#faf8fe" stroke={C.projLine} strokeWidth={1.5} strokeDasharray="5 4" /><text x={1046} y={684}>Training only</text>
          <rect x={1220} y={666} width={36} height={24} rx={5} fill={GOLD.bg} stroke={GOLD.line} strokeWidth={3} /><text x={1266} y={684}>Changed from {ch.from}</text>
          <text x={2130} y={684} textAnchor="end">C: concatenation · LN: LayerNorm · 128-d: shared embedding</text>
        </g>
        <ChangeStrip ch={ch} y={726} />
        {ch.ids.map((id) => <ChangeMark key={id} box={B0_BOX[id]} tag={ch.tag} />)}
        <T x={1086} y={H - 36} s={19}>Single model: no seed ensemble and no ECAPA. C1 × 3 appends ECAPA to the voice input and averages three seeds.</T>
      </svg>
    </div>
  );
}
