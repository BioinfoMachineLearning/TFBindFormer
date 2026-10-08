#!/usr/bin/env python3
"""
Evaluate TFBindFormer on either seen-TF or unseen-TF datasets.

The same evaluation script is used for both settings.

Seen-TF evaluation:
    Evaluate TFs that were included during model training,
    using held-out chromosomes.

Unseen-TF evaluation:
    Evaluate TFs that were completely held out during training
    for zero-shot TF generalization.

Important:
    1. Seen and unseen datasets must use the SAME global
       cell-type/cell-condition ID mapping.

    2. num_cell_types refers to the full embedding vocabulary
       used during training, not only the IDs present in the
       current evaluation dataset.

    3. The protein embeddings, masks, metadata, labels, cached
       pairs, and cell-type IDs must all correspond to the same
       task ordering.

    4. --use_cell_type should only be enabled for checkpoints
       trained with cell-type/cell-condition embeddings.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar


# ============================================================
# Project imports
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.utils import TFBindDataModule
from src.model import LitDNABindingModel


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TFBindFormer on seen-TF or unseen-TF datasets"
        )
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    parser.add_argument(
        "--test_dna_npy",
        type=str,
        required=True,
        help="Path to test DNA .npy file.",
    )

    parser.add_argument(
        "--test_labels_npy",
        type=str,
        required=True,
        help="Path to test label matrix .npy file.",
    )

    parser.add_argument(
        "--test_metadata_tsv",
        type=str,
        required=True,
        help="Path to task metadata TSV file.",
    )

    parser.add_argument(
        "--test_pairs_file",
        type=str,
        required=True,
        help="Path to cached test pairs (.npy or supported format).",
    )

    # --------------------------------------------------------
    # Protein embeddings
    # --------------------------------------------------------

    parser.add_argument(
        "--fixed_tf_embs_pt",
        type=str,
        required=True,
        help="Path to fixed TF embedding tensor.",
    )

    parser.add_argument(
        "--fixed_tf_masks_pt",
        type=str,
        required=True,
        help="Path to fixed TF mask tensor.",
    )

    parser.add_argument(
        "--protein_in_dim",
        type=int,
        default=1024,
        help="Input dimension of the protein embeddings.",
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to trained Lightning checkpoint.",
    )

    # --------------------------------------------------------
    # Cell-type / condition embedding options
    # --------------------------------------------------------

    parser.add_argument(
        "--use_cell_type",
        action="store_true",
        help=(
            "Use cell-type/cell-condition embeddings. "
            "Enable only for checkpoints trained with them."
        ),
    )

    parser.add_argument(
        "--cell_type_dim",
        type=int,
        default=16,
        help="Cell-type embedding dimension used during training.",
    )

    parser.add_argument(
        "--cell_type_ids_npy",
        type=str,
        default=None,
        help=(
            "Optional .npy file containing task-level cell_type_id "
            "values aligned with label columns."
        ),
    )

    # --------------------------------------------------------
    # Evaluation settings
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
    )

    # --------------------------------------------------------
    # Logging / output
    # --------------------------------------------------------

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="tfbind-test",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_out",
    )

    return parser.parse_args()


# ============================================================
# CELL-TYPE ID LOADING
# ============================================================

def load_cell_type_ids(
    args,
    meta,
    test_labels,
):
    """
    Load task-level cell-type/cell-condition IDs.

    Expected shape:
        (num_tasks,)

    The IDs must align with the columns of:
        test_labels[:, task_idx]

    The same global mapping used during training must also be
    used for both seen-TF and unseen-TF evaluation.
    """

    if not args.use_cell_type:

        print(
            "[INFO] Cell-type embeddings disabled."
        )

        return None, 0

    # --------------------------------------------------------
    # Load from external .npy if supplied
    # --------------------------------------------------------

    if args.cell_type_ids_npy is not None:

        print(
            "[INFO] Loading cell_type_ids from:",
            args.cell_type_ids_npy,
        )

        cell_type_ids = np.load(
            args.cell_type_ids_npy
        ).astype(np.int64)

    # --------------------------------------------------------
    # Otherwise use metadata
    # --------------------------------------------------------

    else:

        print(
            "[INFO] Loading cell_type_ids from metadata."
        )

        if "cell_type_id" in meta.columns:

            cell_type_ids = (
                meta["cell_type_id"]
                .values
                .astype(np.int64)
            )

        elif "condition_id" in meta.columns:

            cell_type_ids = (
                meta["condition_id"]
                .values
                .astype(np.int64)
            )

        else:

            raise ValueError(
                "--use_cell_type was enabled, but metadata "
                "contains neither 'cell_type_id' nor "
                "'condition_id', and --cell_type_ids_npy "
                "was not provided."
            )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if cell_type_ids.ndim != 1:

        raise ValueError(
            "cell_type_ids must be a 1D array, but got shape: "
            f"{cell_type_ids.shape}"
        )

    if test_labels.shape[1] != len(cell_type_ids):

        raise ValueError(
            f"Task count mismatch:\n"
            f"  test_labels columns : {test_labels.shape[1]}\n"
            f"  cell_type_ids       : {len(cell_type_ids)}"
        )

    if len(cell_type_ids) == 0:

        raise ValueError(
            "cell_type_ids is empty."
        )

    # --------------------------------------------------------
    # Current dataset fallback only
    # --------------------------------------------------------
    #
    # This is NOT necessarily the embedding vocabulary size
    # used during training.
    #
    # Example:
    #
    #     unseen set max ID = 63
    #
    # but checkpoint may contain:
    #
    #     nn.Embedding(89, 16)
    #
    # Therefore checkpoint information takes priority later.
    # --------------------------------------------------------

    fallback_num_cell_types = (
        int(cell_type_ids.max()) + 1
    )

    print(
        "[INFO] Using cell-type embeddings."
    )

    print(
        "[INFO] cell_type_ids shape:",
        cell_type_ids.shape,
    )

    print(
        "[INFO] Cell IDs present in current dataset:",
        len(np.unique(cell_type_ids)),
    )

    print(
        "[INFO] Current dataset cell ID range:",
        int(cell_type_ids.min()),
        "-",
        int(cell_type_ids.max()),
    )

    print(
        "[INFO] Fallback num_cell_types:",
        fallback_num_cell_types,
    )

    return (
        cell_type_ids,
        fallback_num_cell_types,
    )


# ============================================================
# GET EMBEDDING VOCABULARY FROM CHECKPOINT
# ============================================================

def get_num_cell_types_from_checkpoint(
    ckpt_path,
    fallback_num_cell_types,
):
    """
    Determine the full cell-type/condition embedding vocabulary
    used during training.

    Priority:
        1. checkpoint hyper_parameters["num_cell_types"]
        2. model.cell_type_embedding.weight shape
        3. fallback inferred from current evaluation IDs

    This is important for unseen-TF evaluation because the
    unseen dataset may contain only a subset of the global
    cell-type/condition IDs.
    """

    print(
        "[INFO] Inspecting checkpoint cell-type vocabulary..."
    )

    try:

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        # ----------------------------------------------------
        # Option 1:
        # num_cell_types saved in Lightning hyperparameters
        # ----------------------------------------------------

        hparams = ckpt.get(
            "hyper_parameters",
            {},
        )

        if (
            "num_cell_types" in hparams
            and hparams["num_cell_types"] is not None
        ):

            num_cell_types = int(
                hparams["num_cell_types"]
            )

            print(
                "[INFO] num_cell_types loaded from "
                "checkpoint hyperparameters:",
                num_cell_types,
            )

            return num_cell_types

        # ----------------------------------------------------
        # Option 2:
        # Infer directly from saved embedding table
        # ----------------------------------------------------

        state_dict = ckpt.get(
            "state_dict",
            {},
        )

        possible_keys = [
            "model.cell_type_embedding.weight",
            "cell_type_embedding.weight",
        ]

        for key in possible_keys:

            if key in state_dict:

                weight = state_dict[key]

                if weight.ndim != 2:

                    raise ValueError(
                        f"{key} should be 2D, but got "
                        f"shape {tuple(weight.shape)}"
                    )

                num_cell_types = int(
                    weight.shape[0]
                )

                embedding_dim = int(
                    weight.shape[1]
                )

                print(
                    "[INFO] Found cell-type embedding "
                    "in checkpoint:"
                )

                print(
                    f"[INFO]   key: {key}"
                )

                print(
                    f"[INFO]   shape: "
                    f"{tuple(weight.shape)}"
                )

                print(
                    "[INFO] num_cell_types inferred "
                    "from checkpoint:",
                    num_cell_types,
                )

                print(
                    "[INFO] checkpoint cell_type_dim:",
                    embedding_dim,
                )

                return num_cell_types

        print(
            "[WARN] Cell-type embedding table was not "
            "found in checkpoint state_dict."
        )

    except Exception as e:

        print(
            "[WARN] Could not determine num_cell_types "
            f"from checkpoint: {e}"
        )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    print(
        "[WARN] Falling back to num_cell_types:",
        fallback_num_cell_types,
    )

    return fallback_num_cell_types


# ============================================================
# VALIDATE CELL-TYPE IDS
# ============================================================

def validate_cell_type_ids(
    cell_type_ids,
    num_cell_types,
):
    """
    Check that the evaluation IDs are compatible with the
    checkpoint embedding table.
    """

    if cell_type_ids is None:

        return

    min_id = int(
        cell_type_ids.min()
    )

    max_id = int(
        cell_type_ids.max()
    )

    num_unique = len(
        np.unique(cell_type_ids)
    )

    if min_id < 0:

        raise ValueError(
            f"Invalid negative cell_type_id found: {min_id}"
        )

    if max_id >= num_cell_types:

        raise ValueError(
            "\nCell-type ID is incompatible with checkpoint.\n"
            f"Current dataset maximum ID: {max_id}\n"
            f"Checkpoint vocabulary size: {num_cell_types}\n"
            f"Valid checkpoint IDs: 0-{num_cell_types - 1}\n\n"
            "Make sure seen-TF and unseen-TF datasets use "
            "the SAME global cell-type/condition mapping "
            "used during training."
        )

    print()
    print(
        "============================================"
    )
    print(
        " Cell-type embedding configuration"
    )
    print(
        "============================================"
    )

    print(
        "[INFO] Unique IDs in current dataset:",
        num_unique,
    )

    print(
        "[INFO] Current dataset ID range:",
        min_id,
        "-",
        max_id,
    )

    print(
        "[INFO] Full checkpoint vocabulary:",
        num_cell_types,
    )

    print(
        "============================================"
    )
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    pl.seed_everything(
        42,
        workers=True,
    )

    torch.set_float32_matmul_precision(
        "medium"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DNA + LABELS
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " Loading evaluation dataset"
    )
    print(
        "============================================"
    )

    test_dna = np.load(
        args.test_dna_npy,
        mmap_mode="r",
    )

    test_labels = np.load(
        args.test_labels_npy,
        mmap_mode="r",
    )

    print(
        "[INFO] test_dna:",
        test_dna.shape,
        test_dna.dtype,
    )

    print(
        "[INFO] test_labels:",
        test_labels.shape,
        test_labels.dtype,
    )

    if test_labels.ndim != 2:

        raise ValueError(
            "test_labels must be 2D "
            "(num_DNA_windows, num_tasks), "
            f"but got {test_labels.shape}"
        )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print()
    print(
        "[INFO] Loading metadata:",
        args.test_metadata_tsv,
    )

    meta = pd.read_csv(
        args.test_metadata_tsv,
        sep="\t",
    )

    meta.columns = [
        c.strip()
        for c in meta.columns
    ]

    print(
        "[INFO] metadata rows:",
        len(meta),
    )

    print(
        "[INFO] metadata columns:",
        list(meta.columns),
    )

    tf_col = "TF/DNase/HistoneMark"

    if tf_col not in meta.columns:

        raise ValueError(
            f"Metadata must contain column: {tf_col}"
        )

    tf_names = (
        meta[tf_col]
        .astype(str)
        .tolist()
    )

    if test_labels.shape[1] != len(tf_names):

        raise ValueError(
            "\nTask number mismatch:\n"
            f"  label columns : {test_labels.shape[1]}\n"
            f"  metadata rows : {len(tf_names)}"
        )

    print(
        "[INFO] Number of tasks:",
        len(tf_names),
    )

    print(
        "[INFO] Number of unique TFs:",
        len(set(tf_names)),
    )

    # ========================================================
    # LOAD FIXED TF EMBEDDINGS
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " Loading TF embeddings"
    )
    print(
        "============================================"
    )

    fixed_tf_embs = torch.load(
        args.fixed_tf_embs_pt,
        map_location="cpu",
    )

    fixed_tf_masks = torch.load(
        args.fixed_tf_masks_pt,
        map_location="cpu",
    )

    print(
        "[INFO] fixed_tf_embs:",
        fixed_tf_embs.shape,
        fixed_tf_embs.dtype,
    )

    print(
        "[INFO] fixed_tf_masks:",
        fixed_tf_masks.shape,
        fixed_tf_masks.dtype,
    )

    # --------------------------------------------------------
    # TF/task count checks
    # --------------------------------------------------------

    num_tasks = test_labels.shape[1]

    if fixed_tf_embs.shape[0] != num_tasks:

        raise ValueError(
            "\nTF embedding/task mismatch:\n"
            f"  fixed_tf_embs : {fixed_tf_embs.shape}\n"
            f"  label tasks   : {num_tasks}"
        )

    if fixed_tf_masks.shape[0] != num_tasks:

        raise ValueError(
            "\nTF mask/task mismatch:\n"
            f"  fixed_tf_masks : {fixed_tf_masks.shape}\n"
            f"  label tasks    : {num_tasks}"
        )

    # ========================================================
    # CELL-TYPE / CONDITION IDS
    # ========================================================

    (
        cell_type_ids,
        fallback_num_cell_types,
    ) = load_cell_type_ids(
        args=args,
        meta=meta,
        test_labels=test_labels,
    )

    # --------------------------------------------------------
    # Get full vocabulary from training checkpoint
    # --------------------------------------------------------

    if args.use_cell_type:

        num_cell_types = (
            get_num_cell_types_from_checkpoint(
                ckpt_path=args.ckpt_path,
                fallback_num_cell_types=(
                    fallback_num_cell_types
                ),
            )
        )

        validate_cell_type_ids(
            cell_type_ids=cell_type_ids,
            num_cell_types=num_cell_types,
        )

    else:

        num_cell_types = 0

    # ========================================================
    # DATAMODULE
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " Preparing DataModule"
    )
    print(
        "============================================"
    )

    print(
        "[INFO] test_pairs_file:",
        args.test_pairs_file,
    )

    dm = TFBindDataModule(
        test_dna=test_dna,
        test_labels=test_labels,
        cell_type_ids=cell_type_ids,
        test_pairs_file=args.test_pairs_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup(
        stage="test"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " Loading model checkpoint"
    )
    print(
        "============================================"
    )

    print(
        "[INFO] checkpoint:",
        args.ckpt_path,
    )

    print(
        "[INFO] use_cell_type:",
        args.use_cell_type,
    )

    print(
        "[INFO] num_cell_types:",
        num_cell_types,
    )

    print(
        "[INFO] cell_type_dim:",
        args.cell_type_dim,
    )

    model = (
        LitDNABindingModel.load_from_checkpoint(
            args.ckpt_path,

            protein_in_dim=(
                args.protein_in_dim
            ),

            use_cell_type=(
                args.use_cell_type
            ),

            num_cell_types=(
                num_cell_types
            ),

            cell_type_dim=(
                args.cell_type_dim
            ),

            fixed_tf_embs=(
                fixed_tf_embs
            ),

            fixed_tf_masks=(
                fixed_tf_masks
            ),
            output_dir=args.output_dir,
        )
    )

    model.eval()

    print(
        "[INFO] Model loaded successfully."
    )

    # ========================================================
    # WANDB
    # ========================================================

    wandb_logger = None

    if args.wandb_project:

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=args.output_dir,
            log_model=False,
        )

    # ========================================================
    # TRAINER
    # ========================================================

    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "[INFO] accelerator:",
        accelerator,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=args.precision,
        logger=wandb_logger,
        enable_progress_bar=True,

        callbacks=[
            TQDMProgressBar(
                refresh_rate=500
            )
        ],

        default_root_dir=(
            args.output_dir
        ),

        enable_checkpointing=False,
    )

    # ========================================================
    # RUN TEST
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " Running TEST evaluation"
    )
    print(
        "============================================"
    )
    print()

    trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=None,
    )

    print()
    print(
        "============================================"
    )
    print(
        " TEST COMPLETE"
    )
    print(
        "============================================"
    )
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()



#for seen-TF test

'''
mkdir -p ....../results/seed_42/seenTF_eval

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python eval.py \
  --ckpt_path ....../results/ckpts/epoch=16-val/pr_auc=0.5694-val/loss=0.1102.ckpt \
  --test_dna_npy ....../data/dna_data/test/seen/test_data.npy \
  --test_labels_npy ....../data/dna_data/test/seen/test_labels.npy \
  --test_metadata_tsv ....../data/metadata/seen_tf_metadata.tsv \
  --fixed_tf_embs_pt ....../data/tf_data/fixed_length_200/seen_tf/fixed_tf_embs.pt \
  --fixed_tf_masks_pt ....../data/tf_data/fixed_length_200/seen_tf/fixed_tf_masks.pt \
  --test_pairs_file ....../data/cached_pairs/seed42/seentf/test_pairs.npy \
  --use_cell_type \
  --cell_type_dim 16 \
  --cell_type_ids_npy ....../data/metadata/seen_cell_type_ids.npy \
  --protein_in_dim 1024 \
  --batch_size 1024 \
  --num_workers 6 \
  --precision 16-mixed \
  --wandb_project tfbind_eval \
  --run_name eval_seed42_seenTF \
  --output_dir ....../results/seed_42/seenTF_eval \
  > ....../results/seed_42/seenTF_eval/eval_seenTF.log 2>&1 &

'''

#for unseen-TF test

'''
mkdir -p ....../results/seed_42/unseenTF_eval

CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python eval.py \
  --ckpt_path ....../results/ckpts/epoch=16-val/pr_auc=0.5694-val/loss=0.1102.ckpt \
  --test_dna_npy ....../data/dna_data/test/unseen/test_data.npy \
  --test_labels_npy ....../data/dna_data/test/unseen/test_labels.npy \
  --test_metadata_tsv ....../data/metadata/unseen_tf_metadata.tsv \
  --fixed_tf_embs_pt ....../data/tf_data/fixed_length_200/unseen_tf/fixed_tf_embs.pt \
  --fixed_tf_masks_pt ....../data/tf_data/fixed_length_200/unseen_tf/fixed_tf_masks.pt \
  --test_pairs_file ....../data/cached_pairs/seed42/unseentf/test_pairs.npy \
  --use_cell_type \
  --cell_type_dim 16 \
  --cell_type_ids_npy ....../data/metadata/unseen_cell_type_ids.npy \
  --protein_in_dim 1024 \
  --batch_size 1024 \
  --num_workers 6 \
  --precision 16-mixed \
  --wandb_project tfbind_eval \
  --run_name eval_seed42_unseenTF \
  --output_dir ....../results/seed_42/unseenTF_eval \
  > ....../results/seed_42/unseenTF_eval/eval_unseenTF.log 2>&1 &

'''

