# Related agent frameworks for simulation (literature note, 2026-11-19)

Prof. Rabczuk asked us to look at FermiLink, ALL-FEM and similar agents.
**Basis: arXiv abstract pages and search results only; the full papers have
not been read.** Where an abstract is silent, this note says so. A silent
abstract does not mean the paper lacks the feature.

## Summary table

| system | what it automates | how it checks results (as stated in the abstract) | headline numbers |
|---|---|---|---|
| FermiLink (Meng, ..., Tao E. Li; arXiv 2604.03460, 2026) | agent harness for multi-day simulations in about 50 packages across 9 domains; package knowledge bases kept separate from workflows; runs on coding agents (Codex; the project page also lists Claude Code and Gemini CLI) | agreement with published figures, graded as "high-fidelity" or "qualitative"; no convergence, MMS or error-estimation checks stated | 132 figure-reproduction tasks: 74 reproduced (56.1%), 30 (22.7%) high-fidelity; runs "in the absence of human intervention" |
| ALL-FEM (Deotale, Srinivasan, Tian, Zhang, Vlachos, Gomez; arXiv 2603.21011; CMAME vol. 457, 2026) | FEniCS code generation by fine-tuned LLMs (3B-120B) plus a 9-role multi-agent loop; training corpus of 1000+ "verified" scripts | runtime feedback and debugging; how the corpus was verified is not stated; no convergence, MMS or error-estimation checks stated | 39 held-out benchmarks; best model 71.79% "code-level success" (term not defined in the abstract) |
| PDE-Agents (Adhikari, Noorsumar, Jensen; arXiv 2606.07850, 2026) | natural-language FEM lifecycle; 3 agents with a LangGraph supervisor; a knowledge graph (GraphRAG) of materials, failures and run history | **convergence-rate check**: five mesh levels, L2 error against exact solutions, O(h^2) on heat-equation cases (two of three cases reach O(h^2), per a search summary of the full paper) | 97.8% success over 1,369 runs; 85.4% first try |
| CAE agents study (Shi, Zhang; arXiv 2609.03718, 2026) | compares a single-agent generic harness with specialised multi-agent CAE systems (OpenFOAM etc.) | execution-feedback repair only | FoamBench: single agent 96.4%, multi-agent 88.2%; repair raises 71.8% to 96.4% |
| VFEAgent (Zhang et al.; arXiv 2605.28978, 2026) | FEA from images and text; vision-language multi-agent pipeline; "verification-first" code synthesis | code executability and "physical validity" (method not stated) | no numbers in the abstract |
| FeaGPT (Qi, Xu, Chu; arXiv 2510.21993, 2025) | geometry-mesh-simulation-analysis pipeline with CalculiX; adaptive meshing | "validated" results (method not stated); no convergence study stated | turbocharger cases, 432 airfoil configurations; no accuracy numbers |

## How our framework compares

1. **What is being checked.** These systems answer "does the generated
   simulation run and look right?" They measure code success, figure
   reproduction or execution success. Ours answers a different question:
   "is a numerical reference accurate enough to judge a surrogate?" From the
   abstracts, only PDE-Agents has a solution-verification step, a
   convergence-rate fit against exact solutions on smooth heat problems.
   None of the abstracts mentions:
   - QoI convergence without an exact solution (GCI or multilevel fits);
   - refinement-strategy robustness;
   - singular-order checks;
   - tolerances derived from an accuracy requirement;
   - diagnosis of why a quantity does not converge.
2. **Measuring human intervention.** FermiLink and FeaGPT claim operation
   without human intervention but, per their abstracts, do not measure it.
   Our log-based count (`HUMAN_INTERVENTION.md`, 50 interventions) is a
   first measurement of this kind.
3. **Ideas worth adopting:**
   - FermiLink's separation of package knowledge from workflows, which
     could serve our planned literature agent.
   - PDE-Agents' knowledge graph of past runs and failures, which is close
     to our solved-case repository. It could hold diagnosis outcomes
     (cause, test, result) so the agent learns which investigations paid
     off.
   - ALL-FEM's held-out benchmark set as an evaluation protocol.
4. **Risk shared with all of them.** A generated simulation that runs is
   not a verified one. Our B3 groove stresses ran and looked plausible, yet
   they did not converge at a rate. That is the gap our work addresses.

## Still to do

- Read the full papers, at least FermiLink, ALL-FEM and PDE-Agents, to
  confirm the "not stated" items, especially whether any of them performs
  solution verification.
- Search further for agents that perform verification and validation
  (V&V) or uncertainty quantification on FE references, not just code
  generation.

## Sources

- https://arxiv.org/abs/2604.03460, https://github.com/TaoELi/FermiLink, https://fermilink.org/
- https://arxiv.org/abs/2603.21011, https://www.sciencedirect.com/science/article/abs/pii/S0045782526002586, https://fenics-llm.github.io/
- https://arxiv.org/abs/2606.07850
- https://arxiv.org/abs/2609.03718
- https://arxiv.org/abs/2605.28978
- https://arxiv.org/abs/2510.21993
