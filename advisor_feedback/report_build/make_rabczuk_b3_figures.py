import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Global uniform refinement family (8 points, consistent Ntheta/Nr/Nz pattern except 47040)
elements = [2376, 5760, 10200, 15960, 23520, 33120, 45000, 47040]
max_sxx  = [4.13, 1.80, 0.83, 0.81, 0.737, 0.706, 0.891, 0.705]
avg_sxx  = [0.4349, 0.3845, 0.4306, 0.4357, 0.4325, 0.4354, 0.4345, 0.433]
is_family = [True, True, True, True, True, True, True, False]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

fam_el = [e for e, f in zip(elements, is_family) if f]
fam_max = [m for m, f in zip(max_sxx, is_family) if f]
fam_avg = [a for a, f in zip(avg_sxx, is_family) if f]
sep_el = [e for e, f in zip(elements, is_family) if not f]
sep_max = [m for m, f in zip(max_sxx, is_family) if not f]
sep_avg = [a for a, f in zip(avg_sxx, is_family) if not f]

ax1.plot(fam_el, fam_max, "o-", color="tab:red", label="max|sigma_xx| (refinement family)")
ax1.plot(sep_el, sep_max, "o", color="tab:red", mfc="white", markeredgewidth=2,
         label="max|sigma_xx| (47,040, separate mesh shape)")
ax1.set_xscale("log")
ax1.set_xlabel("elements")
ax1.set_ylabel("max|sigma_xx| in groove region")
ax1.set_title("Peak groove stress: NOT monotonic")
ax1.annotate("breaks monotonicity\n(+26% vs. previous point)", xy=(45000, 0.891),
             xytext=(15000, 2.2), arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.plot(fam_el, fam_avg, "o-", color="tab:blue", label="avg|sigma_xx| (refinement family)")
ax2.plot(sep_el, sep_avg, "o", color="tab:blue", mfc="white", markeredgewidth=2,
         label="avg|sigma_xx| (47,040, separate mesh shape)")
ax2.set_xscale("log")
ax2.set_xlabel("elements")
ax2.set_ylabel("avg|sigma_xx| in groove region (volume-weighted)")
ax2.set_title("Region-average groove stress: stable")
ax2.set_ylim(0.3, 0.5)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("Global quadratic (Q20) mesh refinement, groove region, 8 resolutions (2,376-47,040 elements)")
fig.tight_layout()
fig.savefig("/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/final_figs/fig1_global_refinement.png", dpi=150)
plt.close(fig)

# z_beta sweep at fixed 10,200 elements (Q20)
z_beta = [0.0, 1.0, 2.0, 3.0]
max_zb = [0.826, 0.615, 0.684, 0.783]
avg_zb = [0.4306, 0.4289, 0.4234, 0.4272]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(z_beta, max_zb, "o-", color="tab:red")
ax1.set_xlabel("z_beta (axial clustering strength toward groove)")
ax1.set_ylabel("max|sigma_xx|")
ax1.set_title("Peak groove stress vs. local axial clustering\n(fixed 10,200 Q20 elements): also not monotonic")
ax1.grid(alpha=0.3)

ax2.plot(z_beta, avg_zb, "o-", color="tab:blue")
ax2.set_xlabel("z_beta (axial clustering strength toward groove)")
ax2.set_ylabel("avg|sigma_xx|")
ax2.set_title("Region-average groove stress vs. local axial clustering:\nstable")
ax2.set_ylim(0.4, 0.44)
ax2.grid(alpha=0.3)

fig.suptitle("Local axial (z) clustering sweep, quadratic (Q20) elements, fixed element count")
fig.tight_layout()
fig.savefig("/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/final_figs/fig2_zbeta_sweep.png", dpi=150)
plt.close(fig)

print("done")
