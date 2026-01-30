import numpy as np
import os
import pandas as pd

base_path = ".../TFBindFormer/data/dna_data"

# Label files
paths = {
    "train": os.path.join(base_path, "/train/train_labels.npy"),
    "val":   os.path.join(base_path, "/val/valid_labels.npy"),
    "test":  os.path.join(base_path, "/test/test_labels.npy"),
}

summary_rows = []

for split, fpath in paths.items():
    print(f"Loading {split} labels from: {fpath}")
    labels = np.load(fpath).astype(int)  # matrix: (num_windows, num_tfs)

    num_windows, num_tfs = labels.shape
    total = labels.size
    pos = labels.sum()
    neg = total - pos
    ratio = pos / total * 100

    summary_rows.append({
        "split": split,
        "num_bins": num_windows,
        "num_tfs": num_tfs,
        "positives": pos,
        "negatives": neg,
        "positive_ratio_percent": ratio
    })

# Convert to DataFrame
df_stats = pd.DataFrame(summary_rows)

# Print summary table
print("\n================ DATA SUMMARY ================")
print(df_stats.to_string(index=False))
print("================================================\n")

# Save to local CSV
output_path = "dataset_label_stats.csv"
df_stats.to_csv(output_path, index=False)

print(f"Saved dataset summary to: {output_path}")




