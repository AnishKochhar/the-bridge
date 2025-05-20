# Anish Kochhar, May 2025

""" runs full demo of pipeline, writes to results.md """

import json, os, datetime, random
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
from cluster import select_anchors
from graphs import linearise_triples, sample_biased_paths
from arxiv_search import hybrid_search, DOMAIN_QUERIES
from agents import *

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def main(domainA: str, domainB: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log = lambda msg: print(f"[{ts}] {msg}")
    log(f"- {domainA}  and  {domainB} - ")

    user_approval = 0
    while user_approval != 1:
        a, b = select_anchors(domainA, domainB)
        user_approval = int(input(f"\nSelected Anchors: {a} | {b}\n\tEnter 1 for approval and 0 otherwise: "))

    log(f"Anchors: {a} | {b}")
    paths = sample_biased_paths(source=a, target=b, alpha=0.2, num_waypoints=2, attempts=20, top_k=3)

    for i, path in enumerate(paths, 1):
        path_str = linearise_triples(path)
        print(f"\nPath #{i}: {path_str}\n")

        onto = OntologistAgent().run(path_str); log("Ontologist done")
        dom_jsonA = DomainExpertAgent(domainA).run(onto); log(f"Domain {domainA} done")
        dom_jsonB = DomainExpertAgent(domainB).run(onto); log(f"Domain {domainB} done")
        bridge = BridgeAgent().run(dom_jsonA, dom_jsonB); log("Bridge done")
        critic = CriticAgent().run(bridge); log("Critic done")

        if critic["recommendation"] == "REVISE_DOMAIN":
            log("Critic has output REVISE_DOMAIN, revising from experts")
            # TODO: tweak query with random keyword
            dom_jsonA = DomainExpertAgent(domainA).run(onto)
            dom_jsonB = DomainExpertAgent(domainB).run(onto)
            bridge = BridgeAgent().run(dom_jsonA, dom_jsonB)
            critic = CriticAgent().run(bridge); log("Domain revisions applied")
        elif critic["recommendation"] == "REVISE_BRIDGE":
            log("Critic has output REVISE_BRIDGE, revising from Bridge Agent")
            bridge = BridgeAgent().run(dom_jsonA, dom_jsonB)
            critic = CriticAgent().run(bridge); log("Bridge revised")

        raw = {"path": path_str, "ontologist": onto, "domainA": dom_jsonA, "domainB": dom_jsonB,
               "bridge": bridge, "critic": critic }
        save_path = RESULTS_DIR/f"result_{ts}_path_{i}"
        json.dump(raw, open(f"{save_path}.json", "w"), indent=2)

        md = (
            f"# The Bridge — `{a}` → `{b}`\n\n"
            f"## 1. Selected Domains\n- **Domain A**: {domainA}\n- **Domain B**: {domainB}\n\n"
            f"## 2. Knowledge Graph Path\n```\n{path_str}\n```\n\n"
            f"## 3. Ontologist Hypothesis\n"
            f"- **Hypothesis**: {onto['hypothesis']}\n"
            f"- **Mechanism**: {onto['mechanism']}\n"
            f"- **Assumptions**:\n"
            + "".join(f"  - {a}\n" for a in onto.get("assumptions", [])) + "\n"
            f"## 4. Domain Expert A: {domainA}\n"
            f"{dom_jsonA['elaboration']}\n\n"
            f"**References:**\n"
            + "".join(f"- {ref}\n" for ref in dom_jsonA.get("references", [])) + "\n"
            f"\n## 5. Domain Expert B: {domainB}\n"
            f"{dom_jsonB['elaboration']}\n\n"
            f"**References:**\n"
            + "".join(f"- {ref}\n" for ref in dom_jsonB.get("references", [])) + "\n"
            "\n## 6. Bridge Idea\n"
            f"- **Idea**: {bridge['bridge_idea']}\n"
            f"- **Synergies**: {bridge['synergies']}\n"
            f"- **Challenges**: {bridge['challenges']}\n"
            f"**Fused References:**\n"
            + "".join(f"- {ref}\n" for ref in bridge.get("fused_references", [])) + "\n"
            f"\n## 7. Critic Evaluation\n"
            f"- **Novelty**: {critic['novelty']}\n"
            f"- **Feasibility**: {critic['feasibility']}\n"
            f"- **Recommendation**: {critic['recommendation']}\n"
            f"- **Comments**: {critic['comments']}\n"
        )
        Path(f"{save_path}.md").write_text(md)
        log(f"[{i} / {len(paths)}] Results saved to {save_path}")



if __name__ == "__main__":
    domainA, domainB = random.sample(list(DOMAIN_QUERIES.keys()), k=2)
    main(domainA, domainB)