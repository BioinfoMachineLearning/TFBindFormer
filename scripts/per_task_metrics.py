#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def safe_auc(y_true, y_prob):
    """
    AUROC/AUPRC require both positive and negative labels.
    Return NaN if only one class exists.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser(description="Compute per-TF metrics from saved eval outputs")

    parser.add_argument("--probs_npy", required=True)
    parser.add_argument("--targets_npy", required=True)
    parser.add_argument("--test_pairs_npy", required=True)
    parser.add_argument("--metadata_tsv", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tf_col", type=str, default="TF/DNase/HistoneMark")

    args = parser.parse_args()

    probs = np.load(args.probs_npy)
    targets = np.load(args.targets_npy)
    pairs = np.load(args.test_pairs_npy)

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]

    assert args.tf_col in meta.columns, f"metadata missing column: {args.tf_col}"

    probs = probs.reshape(-1)
    targets = targets.reshape(-1).astype(int)

    assert len(probs) == len(targets), (len(probs), len(targets))
    assert len(probs) == len(pairs), (len(probs), len(pairs))

    tf_indices = pairs[:, 1].astype(int)

    rows = []

    for tf_idx in sorted(np.unique(tf_indices)):
        mask = tf_indices == tf_idx

        y_true = targets[mask]
        y_prob = probs[mask]
        y_pred = (y_prob >= args.threshold).astype(int)

        n_total = len(y_true)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        auroc, auprc = safe_auc(y_true, y_prob)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tf_name = meta.iloc[tf_idx][args.tf_col]

        row = {
            "tf_idx": tf_idx,
            "tf_name": tf_name,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_ratio": n_pos / max(1, n_total),
            "AUROC": auroc,
            "AUPRC": auprc,
            "ACC": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
        }

        # optional metadata columns
        for col in ["Cell Type", "Treatment", "cell_type_id", "condition_id"]:
            if col in meta.columns:
                row[col] = meta.iloc[tf_idx][col]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by AUPRC descending
    df = df.sort_values("AUPRC", ascending=False, na_position="last")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("[INFO] Saved per-TF metrics to:", args.out_csv)
    print(df.head(20).to_string(index=False))

    print("\n========== Summary ==========")
    print("Number of TF/tasks:", len(df))
    print("Mean AUROC:", df["AUROC"].mean(skipna=True))
    print("Mean AUPRC:", df["AUPRC"].mean(skipna=True))
    print("Median AUROC:", df["AUROC"].median(skipna=True))
    print("Median AUPRC:", df["AUPRC"].median(skipna=True))


if __name__ == "__main__":
    main()


#seentf


#unseentf
'''
mkdir -p /bml/ping/tfbind_review/tfbind_csbj/results/unseenTF_eval/per_tf_metrices

nohup python per_task_metrics.py \
  --probs_npy /bml/ping/tfbind_review/tfbind_csbj/results/unseenTF_eval/metrics_data/probs.npy \
  --targets_npy /bml/ping/tfbind_review/tfbind_csbj/results/unseenTF_eval/metrics_data/targets.npy \
  --test_pairs_npy /bml/ping/tfbind_review/tfbind_csbj/data/cached_pairs/seed42/unseentf/test_pairs.npy \
  --metadata_tsv /bml/ping/tfbind_review/tfbind_csbj/data/metadata/unseen_tf_metadata.tsv \
  --out_csv /bml/ping/tfbind_review/tfbind_csbj/results/unseenTF_eval/per_tf_metrices/per_tf_metrics.csv \
  --threshold 0.3 \
  >/bml/ping/tfbind_review/tfbind_csbj/results/unseenTF_eval/per_tf_metrices/per_tf_metrics.log 2>&1 &

'''
