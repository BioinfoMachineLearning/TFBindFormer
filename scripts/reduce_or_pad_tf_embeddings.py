#!/usr/bin/env python3
'''
If TF length > 200:
    reduce to 200 by adaptive pooling

If TF length <= 200:
    keep the real embedding positions
    pad the remaining positions to 200
    use mask=True for padded positions
'''

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


TF_ALIAS_MAP = {
    "c-Fos": "c-Fos",
    "eGFP-FOS": "c-Fos",

    "GATA2": "GATA2",
    "eGFP-GATA2": "GATA2",

    "eGFP-JunB": "eGFP-JunB",
    "JunB": "eGFP-JunB",

    "UBTF": "UBF",
    "UBF": "UBF",

    "HA-E2F1": "E2F1",
    "E2F1": "E2F1",

    "GR": "GR",
    "NR3C1": "GR",

    "REST": "NRSF",
    "NRSF": "NRSF",
}


def canonical_name(tf):
    return TF_ALIAS_MAP.get(tf, tf)


def load_embedding_index(embedding_dir):
    """
    Build mapping:
        canonical TF name -> embedding file path

    Example:
        CTCF_P49711_embedding.pt -> CTCF
        AP-2alpha_P05549_embedding.pt -> AP-2alpha
    """
    emb_index = {}

    for fname in os.listdir(embedding_dir):
        if not fname.endswith("_embedding.pt"):
            continue

        core = fname.replace("_embedding.pt", "")
        tf_raw = core.rsplit("_", 1)[0]   # remove UniProt ID
        tf_canon = canonical_name(tf_raw)

        emb_index[tf_canon] = os.path.join(embedding_dir, fname)

    return emb_index


def reduce_long_embedding(emb, target_len=200, method="avgmax"):
    """
    For long TFs only:
        emb: L × D, where L > target_len
        return: target_len × D
    """
    emb = emb.float()

    # L × D -> 1 × D × L
    x = emb.transpose(0, 1).unsqueeze(0)

    if method == "avg":
        out = F.adaptive_avg_pool1d(x, target_len)

    elif method == "max":
        out = F.adaptive_max_pool1d(x, target_len)

    elif method == "avgmax":
        avg = F.adaptive_avg_pool1d(x, target_len)
        maxp = F.adaptive_max_pool1d(x, target_len)
        out = 0.5 * avg + 0.5 * maxp

    else:
        raise ValueError("method must be avg, max, or avgmax")

    # 1 × D × target_len -> target_len × D
    return out.squeeze(0).transpose(0, 1)


def reduce_or_pad_embedding(emb, target_len=200, method="avgmax"):
    """
    If L > target_len:
        reduce to target_len and mask all False

    If L <= target_len:
        keep original positions, pad to target_len, mask padded positions True
    """
    if emb.ndim == 3:
        emb = emb.squeeze(0)

    if emb.ndim != 2:
        raise ValueError(f"Bad embedding shape: {emb.shape}")

    emb = emb.float()
    L, D = emb.shape

    if L > target_len:
        out = reduce_long_embedding(emb, target_len=target_len, method=method)
        mask = torch.zeros(target_len, dtype=torch.bool)  # all real after reduction

    else:
        out = torch.zeros(target_len, D, dtype=torch.float32)
        out[:L] = emb

        mask = torch.ones(target_len, dtype=torch.bool)
        mask[:L] = False

    return out, mask, L


def main():
    parser = argparse.ArgumentParser(
        description="Reduce long TF embeddings and pad short TF embeddings to fixed length"
    )

    parser.add_argument("--metadata_tsv", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--target_len", type=int, default=200)
    parser.add_argument("--method", default="avgmax", choices=["avg", "max", "avgmax"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--tf_col", default="TF/DNase/HistoneMark")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]

    if args.tf_col not in meta.columns:
        raise ValueError(
            f"Cannot find TF column {args.tf_col}. "
            f"Available columns: {meta.columns.tolist()}"
        )

    tf_names = meta[args.tf_col].tolist()
    emb_index = load_embedding_index(args.embedding_dir)

    reduced_list = []
    mask_list = []
    original_lengths = []
    status_list = []
    missing = []

    for i, tf in enumerate(tf_names):
        tf_canon = canonical_name(tf)

        if tf_canon not in emb_index:
            missing.append((i, tf, tf_canon))
            continue

        emb = torch.load(emb_index[tf_canon], map_location="cpu")

        if emb.ndim == 3:
            emb = emb.squeeze(0)

        original_len = emb.shape[0]

        fixed_emb, mask, L = reduce_or_pad_embedding(
            emb,
            target_len=args.target_len,
            method=args.method,
        )

        if original_len > args.target_len:
            status = "reduced"
        elif original_len < args.target_len:
            status = "padded"
        else:
            status = "unchanged"

        reduced_list.append(fixed_emb)
        mask_list.append(mask)
        original_lengths.append(L)
        status_list.append(status)

        print(
            f"{i}: {tf} | original={tuple(emb.shape)} "
            f"-> fixed={tuple(fixed_emb.shape)} | {status}"
        )

    if missing:
        print("\n[ERROR] Missing embeddings:")
        for row_i, tf, tf_canon in missing:
            print(f"  metadata row {row_i}: {tf} -> {tf_canon}")
        raise ValueError("Some TF embeddings are missing. Fix before continuing.")

    fixed_tensor = torch.stack(reduced_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)
    lengths_tensor = torch.tensor(original_lengths, dtype=torch.long)

    if args.dtype == "float16":
        fixed_tensor = fixed_tensor.half()
    else:
        fixed_tensor = fixed_tensor.float()

    torch.save(fixed_tensor, os.path.join(args.out_dir, "fixed_tf_embs.pt"))
    torch.save(mask_tensor, os.path.join(args.out_dir, "fixed_tf_masks.pt"))
    torch.save(lengths_tensor, os.path.join(args.out_dir, "original_tf_lengths.pt"))

    pd.DataFrame({
        "task_id": np.arange(len(tf_names)),
        "tf_name": tf_names,
        "canonical_tf": [canonical_name(x) for x in tf_names],
        "original_length": original_lengths,
        "status": status_list,
    }).to_csv(
        os.path.join(args.out_dir, "tf_names_in_label_order.tsv"),
        sep="\t",
        index=False,
    )

    print("\n========== Summary ==========")
    print("Output embedding:", fixed_tensor.shape, fixed_tensor.dtype)
    print("Output mask:", mask_tensor.shape, mask_tensor.dtype)
    print("Original min length:", min(original_lengths))
    print("Original max length:", max(original_lengths))
    print(pd.Series(status_list).value_counts())

    print("\nSaved to:")
    print(args.out_dir)


if __name__ == "__main__":
    main()



#for seen tf
'''
python reduce_or_pad_tf_embeddings.py \
  --metadata_tsv /bmlfast/ping/tfbind_review/TFBindFormer/build_data_with_holdoutTFS/cell_type_only/seen_metadata_tfbs_with_celltype.tsv \
  --embedding_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings/tf_embeddings_1024_mean \
  --out_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings/fixed_length_200/seentf \
  --target_len 200 \
  --method avgmax \
  --dtype float16
'''
#for unseen tf
'''
python reduce_or_pad_tf_embeddings.py \
  --metadata_tsv /bmlfast/ping/tfbind_review/TFBindFormer/build_data_with_holdoutTFS/cell_type_only/unseen_metadata_tfbs_with_celltype.tsv \
  --embedding_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings/tf_embeddings_1024_mean \
  --out_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings/fixed_length_200/unseentf \
  --target_len 200 \
  --method avgmax \
  --dtype float16

'''