import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data
# -------------------------
metrics_main = ["AUROC"]
metrics_aupr = ["AUPRC"]

full = [0.956]
no_aa = [0.953]
no_3di = [0.954]

full_aupr = [0.385]
no_aa_aupr = [0.372]
no_3di_aupr = [0.380]

width_auroc = 0.18
width_aupr  = 0.32

# -------------------------
# Font sizes (REDUCED)
# -------------------------
LABEL_FS  = 7
TICK_FS   = 6
LEGEND_FS = 5
ANNOT_FS  = 6

# -------------------------
# Figure (single column)
# -------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(3.0, 2.2),      # small, single-column
    gridspec_kw={"width_ratios": [1, 1]}
)

# ===== Left panel: AUROC =====
x1 = np.arange(len(metrics_main))
ax1.bar(x1 - width_auroc, full,   width_auroc, label="TFBindFormer")
ax1.bar(x1,                no_aa, width_auroc, label="Without AA")
ax1.bar(x1 + width_auroc,  no_3di,width_auroc, label="Without 3Di")

ax1.set_xticks(x1)
ax1.set_xticklabels(metrics_main, fontsize=TICK_FS)
ax1.set_ylabel("Performance", fontsize=LABEL_FS)
ax1.set_ylim(0.90, 1.00)
ax1.tick_params(axis="y", labelsize=TICK_FS)
ax1.grid(axis="y", linestyle="--", alpha=0.25)

# Legend (smaller)
ax1.legend(
    frameon=False,
    fontsize=LEGEND_FS,
    loc="upper left",
    handlelength=1.0,
    labelspacing=0.3
)

# ===== Right panel: AUPR =====
x2 = np.arange(len(metrics_aupr))
ax2.bar(x2 - width_aupr, full_aupr,   width_aupr)
ax2.bar(x2,               no_aa_aupr, width_aupr)
ax2.bar(x2 + width_aupr,  no_3di_aupr,width_aupr)

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_aupr, fontsize=TICK_FS)
ax2.set_ylim(0.35, 0.40)
ax2.tick_params(axis="y", labelsize=TICK_FS)
ax2.grid(axis="y", linestyle="--", alpha=0.25)

# Annotations (smaller text & arrows)
ax2.annotate(
    "−0.013",
    xy=(x2[0], no_aa_aupr[0]),
    xytext=(x2[0], no_aa_aupr[0] + 0.008),
    arrowprops=dict(arrowstyle="->", lw=0.7),
    ha="center",
    fontsize=ANNOT_FS
)

ax2.annotate(
    "−0.005",
    xy=(x2[0] + width_aupr, no_3di_aupr[0]),
    xytext=(x2[0] + width_aupr, no_3di_aupr[0] + 0.006),
    arrowprops=dict(arrowstyle="->", lw=0.7),
    ha="center",
    fontsize=ANNOT_FS
)

# -------------------------
# Save
# -------------------------
plt.tight_layout(pad=0.35)
plt.savefig(
    "tfbindformer_ablation_single_column_final.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
