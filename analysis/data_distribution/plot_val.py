import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter

# ======================
# Load VAL labels
# ======================
labels = np.load(".../TFBindFormer/data/dna_data/val/valid_labels.npy")
pos_per_tf = labels.sum(axis=0)

# ======================
# Compute statistics
# ======================
Q1      = np.percentile(pos_per_tf, 25)
median  = np.median(pos_per_tf)
Q3      = np.percentile(pos_per_tf, 75)
mean    = np.mean(pos_per_tf)
min_val = np.min(pos_per_tf)
max_val = np.max(pos_per_tf)

# ======================
# Plot style
# ======================
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.set_position([0.18, 0.18, 0.66, 0.60])

# ======================
# Violin plot
# ======================
sns.violinplot(
    data=pos_per_tf,
    inner=None,
    cut=0,
    width=0.85,
    color="#6baed6",
    linewidth=1.8,
    ax=ax
)

for artist in ax.findobj(PolyCollection):
    artist.set_edgecolor("black")
    artist.set_linewidth(1.6)

# ======================
# Boxplot overlay
# ======================
sns.boxplot(
    data=pos_per_tf,
    width=0.18,
    boxprops=dict(facecolor="none", edgecolor="purple", linewidth=2.4),
    whiskerprops=dict(color="purple", linewidth=2.4),
    capprops=dict(color="purple", linewidth=2.4),
    medianprops=dict(color="black", linewidth=2.4),
    showfliers=False,
    ax=ax
)

# Mean marker
ax.scatter(0, mean, s=140, color="red", zorder=10)

# ======================
# Reference lines
# ======================
ax.axhline(Q1, linestyle="--", color="purple", linewidth=1.6)
ax.axhline(median, linestyle="--", color="black", linewidth=1.6)
ax.axhline(Q3, linestyle="--", color="purple", linewidth=1.6)

# ======================
# Annotations (moved higher & staggered)
# ======================
x_text = 0.22
base_offset = 0.012 * (max_val - min_val)
label_fs = 11

ax.text(x_text, Q3     + base_offset * 1.4, f"Q3 = {int(Q3):,}", color="purple", fontsize=label_fs)
ax.text(x_text, mean   + base_offset * 1.8, f"Mean = {int(mean):,}", color="red", fontsize=label_fs)
ax.text(x_text, median + base_offset * 1.1, f"Median = {int(median):,}", color="black", fontsize=label_fs)
ax.text(x_text, Q1     + base_offset * 0.8, f"Q1 = {int(Q1):,}", color="purple", fontsize=label_fs)

# ======================
# Y-axis formatting (K only on ticks)
# ======================
def thousands(x, pos):
    return f"{int(x/1000)}K" if x >= 1000 else f"{int(x)}"

ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_ylabel("Number of Positive Bin", fontsize=18)
ax.tick_params(axis="y", labelsize=14)

# ======================
# Titles
# ======================
fig.text(
    0.5, 0.93,
    "Distribution of Positive Bin Count per TF (validation)",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold"
)

fig.text(
    0.5, 0.89,
    f"min = {min_val:,}  •  max = {max_val:,}  (K = 1,000)",
    ha="center",
    va="center",
    fontsize=12
)

# ======================
# Axes cleanup
# ======================
ax.set_xticks([])
ax.margins(y=0.05)

# ======================
# Save
# ======================
plt.savefig("val_violin.png", dpi=400)
plt.show()

print("Saved: val_violin.png")