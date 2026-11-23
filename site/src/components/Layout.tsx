import type { ReactNode } from "react";
import { REPO_URL } from "../data/registry";

/** A numbered page section in the shared project-page style ("01 / THE CHALLENGE"). */
export function Section({ id, n, eyebrow, title, intro, band = false, children }:
  { id: string; n?: string; eyebrow: string; title: string; intro?: ReactNode; band?: boolean; children: ReactNode }) {
  const body = (
    <>
      <p className="kicker">{n ? `${n} / ` : ""}{eyebrow}</p>
      <h2 id={`${id}-title`}>{title}</h2>
      {intro && <p className="lead">{intro}</p>}
      {children}
    </>
  );
  return band ? (
    <section id={id} aria-labelledby={`${id}-title`} className="band"><div className="container section">{body}</div></section>
  ) : (
    <section id={id} aria-labelledby={`${id}-title`} className="container section">{body}</section>
  );
}

export function Footer() {
  return (
    <footer>
      <div className="container">
        <span>Matching Faces and Voices Across Languages · FLAG 2027</span>
        <nav aria-label="Footer">
          <a href={`${REPO_URL}/blob/main/docs/publication/report.md`} target="_blank" rel="noreferrer">Report</a>
          <a href={`${REPO_URL}/blob/main/public_results/experiments.json`} target="_blank" rel="noreferrer">Data</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </div>
    </footer>
  );
}
