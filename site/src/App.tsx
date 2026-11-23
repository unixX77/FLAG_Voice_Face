import { useState } from "react";
import { Footer } from "./components/Layout";
import { Challenge, DataMethod, Hero } from "./components/Intro";
import { ArchitectureExplorer } from "./components/Architecture";
import { Journey, SelectionContext } from "./components/Journey";
import { ExperimentExplorer } from "./components/Explorer";
import { FinalResults, Findings, Limitations, Reproducibility } from "./components/Results";

export default function App() {
  const [selected, select] = useState("g3_cat_farl");
  return (
    <SelectionContext.Provider value={{ selected, select }}>
      <a href="#journey" className="skip">Skip to the research journey</a>
      <Hero />
      <main>
        <Challenge />
        <DataMethod />
        <ArchitectureExplorer />
        <Journey />
        <ExperimentExplorer />
        <Findings />
        <FinalResults />
        <Limitations />
        <Reproducibility />
      </main>
      <Footer />
    </SelectionContext.Provider>
  );
}
