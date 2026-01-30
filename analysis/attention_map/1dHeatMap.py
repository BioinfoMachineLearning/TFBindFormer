import numpy as np
import os
import matplotlib.pyplot as plt

# ======================
# Match ablation figure size & fonts
# ======================
FIG_W, FIG_H = 3.0, 2.2

AXIS_LABEL_FS = 7
TICK_LABEL_FS = 6
LEGEND_FS     = 5
ANNOT_FS      = 6

attn_path = ".../heatmap/outputs/attn_weights/tf_to_dna_attn.npy"
fig_dir   = ".../heatmap/outputs/figures_1d"
os.makedirs(fig_dir, exist_ok=True)

# ======================
# Load attention weights
# ======================
tf_dna = np.load(attn_path)   # (B, 200, 200)

pos_idx = 3   # bound
neg_idx = 5   # unbound

tf_dna_pos = tf_dna[pos_idx].mean(axis=0)
tf_dna_neg = tf_dna[neg_idx].mean(axis=0)

# ======================
# Core-restricted peak stats
# ======================
core_start, core_end = 80, 120

core_pos = tf_dna_pos[core_start:core_end]
core_neg = tf_dna_neg[core_start:core_end]

peak_pos = core_pos.max()
peak_neg = core_neg.max()

peak_pos_idx = core_start + core_pos.argmax()
peak_neg_idx = core_start + core_neg.argmax()

# ======================
# Plot
# ======================
x = np.arange(200)

plt.figure(figsize=(FIG_W, FIG_H))
ax = plt.gca()

ax.plot(x, tf_dna_pos, label="Bound", linewidth=1.1)
ax.plot(x, tf_dna_neg, label="Unbound", linestyle="--", linewidth=1.1)

# Core region
ax.axvspan(core_start, core_end, alpha=0.18)
ax.axvline(core_start, linestyle=":", linewidth=0.6)
ax.axvline(core_end,   linestyle=":", linewidth=0.6)

# ======================
# Peak annotations
# ======================
ax.plot(peak_pos_idx, peak_pos, marker="o", markersize=2.6, zorder=5)

ax.annotate(
    "",
    xy=(peak_pos_idx, peak_pos),
    xytext=(core_start - 14, peak_pos * 0.90),
    arrowprops=dict(arrowstyle="-|>", lw=0.6)
)

ax.text(
    core_start - 16,
    peak_pos * 0.88,
    f"Peak = {peak_pos:.3f}\n(token {peak_pos_idx})",
    fontsize=ANNOT_FS,
    ha="right",
    va="top"
)

ax.plot(peak_neg_idx, peak_neg, marker="o", markersize=2.6, zorder=5)

ax.annotate(
    "",
    xy=(peak_neg_idx, peak_neg),
    xytext=(peak_neg_idx - 12, peak_neg * 1.45),
    arrowprops=dict(arrowstyle="-|>", lw=0.6)
)

ax.text(
    peak_neg_idx - 14,
    peak_neg * 1.47,
    f"Peak = {peak_neg:.3f}\n(token {peak_neg_idx})",
    fontsize=ANNOT_FS,
    ha="right",
    va="bottom"
)

# ======================
# Axes, ticks, legend
# ======================
ax.set_xlabel("DNA token position", fontsize=AXIS_LABEL_FS)
ax.set_ylabel("Mean TF→DNA attention", fontsize=AXIS_LABEL_FS)

ax.set_xticks(np.arange(0, 201, 40))
ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)

ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper right")

# Headroom
ax.set_ylim(top=ax.get_ylim()[1] * 1.05)

plt.tight_layout(pad=0.35)

# ======================
# Save
# ======================
out_png = os.path.join(fig_dir, "heatmaps.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print("Saved PNG:", out_png)
