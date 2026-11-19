# Related agent frameworks for simulation (literature note, 2026-11-19)

Prof. Rabczuk asked us to look at FermiLink, ALL-FEM and similar agents.
**Basis:** the table below comes from arXiv abstracts. FermiLink, ALL-FEM
and PDE-Agents were then read in full text (section "Full-text reading"
below). In that reading, parts of ALL-FEM (Sec. 2.2.4 onward, the
appendices) and of PDE-Agents (Sec. 4-9) came through an extracted summary
rather than verbatim text; this is marked there. The other three papers
were read at abstract level only.

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

## Full-text reading (FermiLink, ALL-FEM, PDE-Agents), 2026-11-19

| question | FermiLink (2604.03460 v3) | ALL-FEM (2603.21011 v2) | PDE-Agents (2606.07850 v2) |
|---|---|---|---|
| How results are checked | against the published figure; quality rated by two human evaluators, a third if needed (Sec. 3.1). No convergence study, MMS, a-posteriori error or UQ; "always provide uncertainty information" (Sec. 2) without a method | compared with expert reference solutions by plot overlay, then an LLM judge, differences checked by hand (Sec. 3.1*); a relative-L2 check against a finer mesh appears only as an alternative (App. 6*). No convergence study by the agent | a V&V study "following ASME V&V 10-2006" with five mesh levels and log-log rate fits (Sec. 5*), reported by the authors; the agent itself runs nine rule-based pre-run checks, none on convergence (Sec. 4) |
| What counts as success | Reproduced / Replotted / Blocked; reproduced tasks graded by humans; no numeric tolerance | match with the reference solution (visual, LLM judge) after the code runs | completion plus a "physics score" (T_max/T_min in expected ranges); notes that "a simulation with fabricated properties still 'succeeds' computationally" (Sec. 6.1*) |
| Failure diagnosis | the agent rejected its own first fit and refit (Sec. 3.3); warns against "shortcut-seeking" and calls for "process-level validation" | Corrector reads code + run-time error (Sec. 2.3.2); numerical errors in code that RUNS are not caught* | ReAct loop with a debug tool; mostly boundary-condition or CFL fixes (Sec. 7.3*) |
| Human involvement | benchmark with "zero human expert insights"; a separate expert-guided set; interventions not counted; human expertise "is needed" to set objectives and judge validity (Sec. 4) | a human Admin agent must stop the loop*; not counted | not counted |
| Solid mechanics / hyperelasticity | none (Psi4, CP2K, MEEP, LAMMPS, ...) | linear/nonlinear elasticity, hyperelasticity, plasticity in the benchmark; no hyperelastic-specific validation found | none: heat equation only (Sec. 8*) |
| Memory of past runs | short/long-term memory of setups and pitfalls (Sec. 2) | none | Neo4j knowledge graph; top-3 similar past runs injected as examples; +0.08 physics score (Sec. 6.3) |

\* = from the extracted summary of that part of the paper, not verbatim text.

Conclusion after the full text: none of the three verifies a numerical
REFERENCE the way our framework does. Two of them rely on comparison
with a known answer (published figure, expert reference): FermiLink with
human grading, ALL-FEM with a visual check and an LLM judge. PDE-Agents
fits convergence rates, but the authors do it outside the agent and only
for heat-equation cases with exact solutions. None performs QoI convergence
without an exact solution, strategy robustness, singular-order checks or a
non-convergence diagnosis, and none counts human interventions. Two
statements support our approach directly: FermiLink's call for
"process-level validation" and PDE-Agents' remark that a run with
fabricated inputs still "succeeds".

## Still to do

- Read the remaining parts of ALL-FEM (appendices) and PDE-Agents
  (Sec. 4-9) verbatim; read VFEAgent, FeaGPT and the CAE-harness study in
  full.
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
