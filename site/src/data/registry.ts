// Typed access to the canonical experiment registry (public_results/experiments.json).
// Components never hard-code metric values: they look them up here by experiment id.
import raw from "./experiments.json";

export type Metrics = Record<string, number>;
export interface Delta { value: number; ci95?: [number, number]; p_b_better?: number }
export interface Source { path: string; type: string; lines?: string; extract?: string }
export interface Experiment {
  id: string;
  phase: string;
  family: string;
  title: string;
  question: string;
  hypothesis?: string;
  protocol: string;
  outcome: "adopted" | "rejected" | "diagnostic" | "candidate" | "baseline" | "official";
  interpretation: string;
  seeds?: number[];
  languages?: string[];
  systems?: Record<string, Metrics>;
  cells?: Record<string, Metrics>;
  comparison?: { a: string; b: string; seeds?: number[]; delta: Record<string, Delta> };
  per_fold?: Record<string, Record<string, number[]>>;
  deltas?: Record<string, Record<string, Delta>>;
  unimodal?: Record<string, Metrics>;
  recovery?: Record<string, number | null>;
  points?: Array<Record<string, number>>;
  systems_urdu?: Record<string, Metrics>;
  systems_german?: Record<string, Metrics>;
  ensemble_vs_single?: Record<string, Record<string, Delta>>;
  term_weights?: Record<string, Record<string, number>>;
  context?: Record<string, Record<string, number | string>>;
  sources: Source[];
}
interface Registry {
  units: string;
  protocols: Record<string, string>;
  experiments: Experiment[];
  score_histograms: {
    system: string; protocol: string; score_type: string; pooled_over: string; source: string;
    tracks: Record<"standard" | "gender", { bins: number[]; genuine: number[]; impostor: number[]; eer: number }>;
  };
}

export const registry = raw as unknown as Registry;
export const experiments = registry.experiments;
const byId = new Map(experiments.map((e) => [e.id, e]));

export function exp(id: string): Experiment {
  const e = byId.get(id);
  if (!e) throw new Error(`Unknown experiment id: ${id}`);
  return e;
}

/** A metric value from an experiment; throws if absent so missing data is never shown as a number. */
export function metric(id: string, system: string, key: string, field: "systems" | "systems_urdu" | "systems_german" | "cells" | "unimodal" = "systems"): number {
  const table = exp(id)[field] as Record<string, Metrics> | undefined;
  const v = table?.[system]?.[key];
  if (typeof v !== "number") throw new Error(`Missing ${field}.${system}.${key} in ${id}`);
  return v;
}

export function delta(id: string, key: string): Delta {
  const d = exp(id).comparison?.delta?.[key];
  if (!d) throw new Error(`Missing comparison.delta.${key} in ${id}`);
  return d;
}

export function transferDelta(id: string, pair: string, key: string): Delta {
  const d = exp(id).deltas?.[pair]?.[key];
  if (!d) throw new Error(`Missing deltas['${pair}'].${key} in ${id}`);
  return d;
}

export const fmt = (v: number, digits = 2) => v.toFixed(digits);
export const fmtDelta = (v: number, digits = 2) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(digits)}`;
export const fmtCI = (ci?: [number, number]) => (ci ? `[${fmtDelta(ci[0])}, ${fmtDelta(ci[1])}]` : "");

export const PROTOCOL_LABEL: Record<string, string> = {
  codabench_dev: "Official · CodaBench dev",
  codabench_eval: "Official · CodaBench evaluation",
  local_5fold: "Local · 5-fold (English)",
  local_5fold_terms: "Local · 5-fold + score terms",
  urdu_holdout: "Holdout · Urdu (unseen)",
  urdu_holdout_seedavg: "Holdout · Urdu (seed-averaged)",
  german_holdout: "Holdout · German (unseen)",
  unimodal_diagnostic: "Diagnostic · unimodal",
  simulation: "Simulation · hidden labels",
};

export const OUTCOME_STYLE: Record<Experiment["outcome"], string> = {
  adopted: "chip-adopted", candidate: "chip-candidate", baseline: "chip-baseline",
  official: "chip-official", diagnostic: "chip-diagnostic", rejected: "chip-rejected",
};

export const REPO_URL = "https://github.com/unixX77/FLAG_Voice_Face";
export const sourceUrl = (path: string, lines?: string) => {
  const base = `${REPO_URL}/blob/main/${path}`;
  if (!lines) return base;
  const [a, b] = lines.split("-");
  return `${base}#L${a}${b ? `-L${b}` : ""}`;
};
