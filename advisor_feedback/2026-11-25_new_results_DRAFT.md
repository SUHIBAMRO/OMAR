# New results - DRAFT (2026-11-25), generated from the stored runs

Not sent. Every line below is written by campaign.py from autonomous_runs/*/run.json.

| run | status | dim | material | code verification | certified | open | person points |
|---|---|---|---|---|---|---|---|
| 1910.09459 | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| cook_3d | stopped at V | 3D | linear_elastic | Q2 INCONCLUSIVE, P2 INCONCLUSIVE | 0 () | (not reached) | 2 |
| cook_linear | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 4 (corner0_gk_avg_von_mises, corner2_gk_avg_von_mises, strain_energy, v_C) | corner1_gk_avg_von_mises, peak_von_mises | 4 |
| deep_ritz_2305.15257 | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| nh_l_bracket_demo | complete | 2D | neo_hookean | Q2 PASS, P2 PASS | 0 () | corner0_gk_avg_von_mises, corner1_gk_avg_von_mises, corner2_gk_avg_von_mises, peak_von_mises, strain_energy, u_top_left, v_right_top | 1 |
| nh_plate_hole_3d_pe | stopped at V | 3D | neo_hookean | Q2 INCONCLUSIVE, P2 PASS | 0 () | (not reached) | 2 |
| nh_plate_hole_pe | complete | 2D | neo_hookean | Q2 PASS, P2 PASS | 3 (peak_von_mises, strain_energy, u_right) | v_hole_top | 1 |
| plate_hole_quarter | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 4 (peak_von_mises, strain_energy, u_right, v_hole_top) |  | 5 |
| plate_hole_quarter_3d_pe | complete | 3D | linear_elastic | Q2 PASS, P2 PASS | 2 (strain_energy, u_right) | peak_von_mises, v_hole_top | 2 |
| plate_hole_quarter_pe | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 4 (peak_von_mises, strain_energy, u_right, v_hole_top) |  | 1 |
| polivem_2609.35878 | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| polivem_2609.35878_posthoc | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| qtt_fem_2501.07778 | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| qtt_fem_2501.07778_posthoc | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| t4_step_plate | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 6 (corner0_gk_avg_von_mises, corner1_gk_avg_von_mises, corner3_gk_avg_von_mises, strain_energy, u_tip_top, v_tip_bottom) | corner2_gk_avg_von_mises, peak_von_mises | 2 |
| t5_quarter_cutout | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 3 (strain_energy, u_right_bottom, v_right_top) | corner0_gk_avg_von_mises, peak_von_mises | 2 |
| t6_cantilever_parabolic | stopped at D (unsupported) | 2D | linear_elastic | - | 0 () | (not reached) | 1 |
| t7_missing_modulus | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 4 (corner0_gk_avg_von_mises, corner1_gk_avg_von_mises, strain_energy, v_top_left) | corner2_gk_avg_von_mises, peak_von_mises | 2 |
| t_bracket | complete | 2D | linear_elastic | Q2 PASS, P2 PASS | 7 (corner0_gk_avg_von_mises, corner1_gk_avg_von_mises, corner2_gk_avg_von_mises, corner3_gk_avg_von_mises, strain_energy, u_left_top, v_left_tip) | peak_von_mises | 1 |

## Questions for Professor Rabczuk

1. When should the agent ask a person to confirm an assumed input? The pre-registered rule (the input changes a certified quantity by more than its numerical uncertainty) asks for every input on smooth problems; the post hoc proposal asks only when the change exceeds the adopted accuracy target. (evidence: sensitivity_runs/*.json, sensitivity_preregistration.json)
2. Should a quantity certified at an earlier stage of a run keep its certificate when a later state (more levels) no longer certifies it? Today the loop keeps it. (evidence: autonomous_runs/plate_hole_quarter/report.pdf (u_right))
3. May code verification on a reference domain (same code paths, regular meshes) replace an INCONCLUSIVE check on the production domain (rules R1/R2, post hoc)? (evidence: round_q_preregistration.json; autonomous_runs/cook_3d/report.pdf)

## Runs waiting for the cluster

- cook_3d stopped: production check INCONCLUSIVE and the reference check did not pass (reference INCONCLUSIVE): no solution verification; cluster job mms_cook_3d_Q2_P2_n32 prepared
- nh_plate_hole_3d_pe stopped: production check INCONCLUSIVE and the reference check did not pass (reference INCONCLUSIVE): no solution verification; cluster job mms_nh_plate_hole_3d_pe_Q2_n32 prepared
