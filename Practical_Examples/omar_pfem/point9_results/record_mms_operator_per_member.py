"""Record the MMS operator's per-test-family-member re-score (from
Round6_MMS_Operator_PerMember.ipynb's Colab stdout) as a committed data
file, and compare its consistency against Q4's.

This closes PROJECT_STATUS.md's open question under point 9: "is the
operator consistent across the family, or merely consistent on average?"
mms_operator*.json only ever stored the family MEAN; mms_operator_per_member
re-scores the same three checkpoints (no retraining) and keeps every
member's row. The means below are asserted equal to the family sweep's
`operator_family_mean` to the printed precision -- same model, same 16
members, same computation, so they must match exactly, and that agreement
is the check that the re-scoring did not drift from the original run.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAM = json.load(open(os.path.join(HERE, 'mms_family_fem_B1_neo_hookean.json')))
FROWS = {r['N']: r for r in FAM['rows']}
METRICS = ('L2_rel', 'H1_semi_rel', 'stress_rel_L2', 'energy_rel')

# Transcribed from the Colab stdout of Round6_MMS_Operator_PerMember.ipynb,
# commit 05cdfa1, run on an A100. Printed precision (4-5 significant
# figures), same convention as every other transcribed run in this
# directory -- the run's own JSON is on Drive at
# pfem_run/mms/operator_per_member/mms_operator_per_member_N{N}.json.
PER_MEMBER_SUMMARY = {
    9: {
        'L2_rel': {'mean': 8.3792e-03, 'std': 3.2189e-03, 'std_over_mean': 0.3842, 'min': 4.2484e-03, 'max': 1.4110e-02},
        'H1_semi_rel': {'mean': 1.1456e-01, 'std': 3.8916e-04, 'std_over_mean': 0.0034, 'min': 1.1409e-01, 'max': 1.1582e-01},
        'stress_rel_L2': {'mean': 1.1593e-01, 'std': 7.2017e-04, 'std_over_mean': 0.0062, 'min': 1.1512e-01, 'max': 1.1743e-01},
        'energy_rel': {'mean': 1.0151e-02, 'std': 5.5755e-03, 'std_over_mean': 0.5492, 'min': 1.2663e-03, 'max': 1.8086e-02},
    },
    17: {
        'L2_rel': {'mean': 8.8261e-03, 'std': 2.2769e-03, 'std_over_mean': 0.2580, 'min': 5.6930e-03, 'max': 1.3705e-02},
        'H1_semi_rel': {'mean': 5.8712e-02, 'std': 9.4627e-04, 'std_over_mean': 0.0161, 'min': 5.7399e-02, 'max': 6.0547e-02},
        'stress_rel_L2': {'mean': 5.9368e-02, 'std': 9.0292e-04, 'std_over_mean': 0.0152, 'min': 5.8172e-02, 'max': 6.1168e-02},
        'energy_rel': {'mean': 9.5955e-03, 'std': 3.5463e-03, 'std_over_mean': 0.3696, 'min': 3.3458e-03, 'max': 1.7160e-02},
    },
    33: {
        'L2_rel': {'mean': 1.2358e-02, 'std': 5.2572e-03, 'std_over_mean': 0.4254, 'min': 6.1579e-03, 'max': 2.5022e-02},
        'H1_semi_rel': {'mean': 4.0947e-02, 'std': 5.4941e-03, 'std_over_mean': 0.1342, 'min': 3.4288e-02, 'max': 5.7564e-02},
        'stress_rel_L2': {'mean': 4.0003e-02, 'std': 4.7889e-03, 'std_over_mean': 0.1197, 'min': 3.4636e-02, 'max': 5.4916e-02},
        'energy_rel': {'mean': 6.4063e-03, 'std': 4.7138e-03, 'std_over_mean': 0.7358, 'min': 1.4792e-04, 'max': 1.8984e-02},
    },
}
NS = sorted(PER_MEMBER_SUMMARY)
assert NS == [9, 17, 33], NS

# The check that matters: this is a RE-SCORE of the same checkpoint on the
# same 16 members mms_family_fem already computed the mean of. If these
# means disagree, the re-scoring script rebuilt a different family or
# loaded the wrong checkpoint.
for N in NS:
    for m in METRICS:
        got = PER_MEMBER_SUMMARY[N][m]['mean']
        want = FROWS[N]['operator_family_mean'][m]
        assert math.isclose(got, want, rel_tol=2e-4), (
            f'N={N} {m}: re-scored mean {got} does not match the family '
            f'sweep\'s {want} -- these must be the same computation')

Q4 = {N: FROWS[N]['Q4_spread_stdev_over_mean'] for N in NS}

# The finding: operator std/mean vs Q4's, per metric, per mesh.
RATIO = {N: {m: PER_MEMBER_SUMMARY[N][m]['std_over_mean'] / max(Q4[N][m], 1e-9)
             for m in METRICS} for N in NS}
# H1/stress start close to Q4's own spread at the coarsest mesh (the
# ceiling-proximity effect Table 24d already established) and grow away
# from it with refinement; L2/energy are inconsistent at every mesh.
assert PER_MEMBER_SUMMARY[9]['H1_semi_rel']['std_over_mean'] < 0.01
assert PER_MEMBER_SUMMARY[33]['H1_semi_rel']['std_over_mean'] > 0.1
for N in NS:
    assert PER_MEMBER_SUMMARY[N]['L2_rel']['std_over_mean'] > 0.2
    assert PER_MEMBER_SUMMARY[N]['energy_rel']['std_over_mean'] > 0.3

print('operator std/mean vs Q4 std/mean, by mesh:')
for N in NS:
    print(f'  N={N:>3}  ' + '  '.join(
        f'{m}: {PER_MEMBER_SUMMARY[N][m]["std_over_mean"]:.4f} '
        f'(Q4 {Q4[N][m]:.4f})' for m in METRICS))

report = {
    'study': ('MMS operator, per-test-family-member re-score of the three '
              'already-trained checkpoints (N=9, 17, 33) -- no retraining. '
              'Closes the open question in PROJECT_STATUS.md under point 9: '
              'is the operator consistent across the family, or merely '
              'consistent on average?'),
    'geometry': 'B1', 'material': 'neo_hookean',
    'checkpoints': {
        9: 'pfem_run/mms/operator_rate/N9/model_best.pt',
        17: 'pfem_run/mms/operator_N17/model_best.pt',
        33: 'pfem_run/mms/operator_rate/N33/model_best.pt',
    },
    'per_member_summary': PER_MEMBER_SUMMARY,
    'q4_std_over_mean_same_family': Q4,
    'operator_over_q4_std_over_mean_ratio': RATIO,
    'mean_agreement_with_family_sweep': (
        'asserted equal (rel_tol 2e-4) to operator_family_mean in '
        'mms_family_fem_B1_neo_hookean.json for all 4 metrics at all 3 '
        'meshes before this file was written -- same checkpoints, same 16 '
        'members, same computation'),
    'reading': {
        'headline': (
            'The operator is not consistent across the family the way Q4 '
            'is, in any norm, at any mesh -- and in L2 and energy it is not '
            'even close. Q4\'s std/mean is 0.000-0.007 across all four '
            'metrics and all three meshes: essentially member-independent. '
            'The operator\'s is 0.26-0.43 in L2 and 0.37-0.74 in energy at '
            'every mesh -- two orders of magnitude more variable.'),
        'h1_and_stress_degrade_with_refinement': (
            'H1 and stress start close to Q4\'s own tight spread at the '
            'coarsest mesh (std/mean 0.003-0.006 at N=9, against Q4\'s '
            '0.000-0.007 -- indistinguishable) and grow away from it as the '
            'mesh refines: 0.015-0.016 at N=17, then 0.12-0.13 at N=33. This '
            'is the same ceiling-proximity effect Table 24d reports for the '
            'MEAN ratio (operator/Q4 stays near 1 in H1/stress at N=9 and '
            'drifts to 1.4x by N=33) -- now shown to affect the operator\'s '
            'per-member RELIABILITY too, not just its average accuracy. '
            'When the operator sits close to the Q4 optimum it inherits '
            'some of Q4\'s member-independence; as training-optimization '
            'error comes to dominate at finer meshes, both the mean error '
            'and its spread across the family grow together.'),
        'l2_and_energy_never_close': (
            'L2 and energy std/mean do not improve at the coarsest mesh the '
            'way H1/stress do -- they are already 0.38-0.55x at N=9, far '
            'above Q4\'s 0.002-0.003. These are the two metrics Table 24d '
            'already showed diverging outright in the mean; here they are '
            'also the two that are inconsistent member-to-member at every '
            'mesh tested, not only at the finest.'),
        'what_this_does_not_establish': (
            'Only B1 x Neo-Hookean, only three meshes, only one operator '
            'checkpoint per mesh (no repeated-seed training to separate '
            'model-to-model variance from family-member variance). And this '
            'is a property of THIS trained network, not a theoretical bound '
            '-- a longer training budget might narrow the spread without '
            'changing the mean, or might not.'),
    },
    'provenance': ('transcribed from the Colab stdout of '
                   'Round6_MMS_Operator_PerMember.ipynb, commit 05cdfa1, run '
                   '2026-09-01 on an A100. The run\'s own JSONs are on Drive '
                   'at pfem_run/mms/operator_per_member/'
                   'mms_operator_per_member_N{9,17,33}.json.'),
}

out = os.path.join(HERE, 'mms_operator_per_member_B1_neo_hookean.json')
with open(out, 'w') as fh:
    json.dump(report, fh, indent=2)
print(f'\nwrote {out}')
