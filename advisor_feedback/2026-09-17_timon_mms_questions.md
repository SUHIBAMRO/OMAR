# Timon's follow-up questions on the MMS force-field normalization/resolution invariance, received 2026-09-17

In reply to the round-11 points 1-3 email (already sent). Verbatim.

---

Dear Omar,

Sound perfect. Just two small points to be sure:

1. For the normalization of the force field passed to the network, are the mean and standard deviation fixed from the training set? This is important as a sample-wise normalization could partly remove information about the absolute magnitude of the loading unless this scale is provided separately.

2. There is also a subtle point about resolution invariance. The consistent assembled nodal force vector itself depends on the resolution since it represents the integral of the body force over the nodal support. For a comparison at one fixed resolution this is perfectly fine. However, if we use the MMS also to discuss coarse-to-fine resolution generalization, we need the continuous body-force field as the spatial input to the network, while continuing to use the resolution-dependent consistent nodal force vector in the energy loss. Otherwise, when changing the resolution, we change not only the discretization but also the numerical representation and magnitude of the network input.

Best regards,
Timon
