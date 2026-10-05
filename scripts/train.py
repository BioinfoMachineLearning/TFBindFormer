#!/usr/bin/env python3
"""
Train GLOBAL TF–DNA Binding Predictor
"""

import os, sys, random, argparse, gc
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.callbacks import TQDMProgressBar

# local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import TFBindDataModule
from src.model import LitDNABindingModel


########################################
# Deterministic Behavior
########################################
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)


gc.collect()
torch.cuda.empty_cache()


###########################################################
# ARGUMENT PARSER
###########################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train GLOBAL TF-DNA Binding Predictor")

    # dataset paths
    parser.add_argument("--train_dna_npy", type=str, required=True)
    parser.add_argument("--train_labels_npy", type=str, required=True)
    parser.add_argument("--train_metadata_tsv", type=str, required=True)

    parser.add_argument("--val_dna_npy", type=str, required=True)
    parser.add_argument("--val_labels_npy", type=str, required=True)
    parser.add_argument("--val_metadata_tsv", type=str, required=True)

    parser.add_argument("--test_dna_npy", type=str, default=None)
    parser.add_argument("--test_labels_npy", type=str, default=None)
    parser.add_argument("--test_metadata_tsv", type=str, default=None)

    #added for loading fixed TF embeddings and masks (if using)
    parser.add_argument("--fixed_tf_embs_pt", type=str, required=True)
    parser.add_argument("--fixed_tf_masks_pt", type=str, required=True)

    parser.add_argument("--train_pairs_file", type=str, required=True)
    parser.add_argument("--val_pairs_file", type=str, required=True)

    parser.add_argument("--protein_in_dim", type=int, default=1024)

    

    # training parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--neg_fraction", type=float, default=0.001)

    # logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="tfbind-global")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--use_cell_type", action="store_true")
    parser.add_argument("--cell_type_dim", type=int, default=16)
    parser.add_argument("--cell_type_ids_npy", type=str, default=None)

    return parser.parse_args()


###########################################################
# MAIN
###########################################################
def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Load datasets (.npy)
    # -------------------------------------------------------
    print("[INFO] Loading .npy datasets...")

    train_dna = np.load(args.train_dna_npy, mmap_mode="r")
    train_labels = np.load(args.train_labels_npy, mmap_mode="r")

    val_dna = np.load(args.val_dna_npy, mmap_mode="r")
    val_labels = np.load(args.val_labels_npy, mmap_mode="r")

    #test_dna = np.load(args.test_dna_npy, mmap_mode="r") if args.test_dna_npy else None
    #test_labels = np.load(args.test_labels_npy, mmap_mode="r") if args.test_labels_npy else None


    # -------------------------------------------------------
    # Load TF names & embeddings
    # -------------------------------------------------------
    print("[INFO] Loading fixed TF embeddings...")

    fixed_tf_embs = torch.load(args.fixed_tf_embs_pt, map_location="cpu")
    fixed_tf_masks = torch.load(args.fixed_tf_masks_pt, map_location="cpu")

    print("fixed_tf_embs:", fixed_tf_embs.shape, fixed_tf_embs.dtype)
    print("fixed_tf_masks:", fixed_tf_masks.shape, fixed_tf_masks.dtype)

    assert fixed_tf_embs.shape[0] == train_labels.shape[1], (
        fixed_tf_embs.shape,
        train_labels.shape,
    )

    assert fixed_tf_masks.shape[0] == train_labels.shape[1], (
        fixed_tf_masks.shape,
        train_labels.shape,
    )

    #load cell type IDs if using cell type embeddings
    cell_type_ids = None
    num_cell_types = 0

    if args.use_cell_type:
        if args.cell_type_ids_npy is None:
            raise ValueError("--cell_type_ids_npy is required when --use_cell_type is used.")

        cell_type_ids = np.load(args.cell_type_ids_npy).astype(np.int64)

        assert len(cell_type_ids) == train_labels.shape[1], (
            len(cell_type_ids),
            train_labels.shape,
        )

        num_cell_types = int(cell_type_ids.max()) + 1

        print("[INFO] Using cell type embeddings")
        print("[INFO] cell_type_ids:", cell_type_ids.shape)
        print("[INFO] num_cell_types:", num_cell_types)

    # -------------------------------------------------------
    # DataModule
    # -------------------------------------------------------
    
    dm = TFBindDataModule(
        train_dna=train_dna,
        train_labels=train_labels,
        val_dna=val_dna,
        val_labels=val_labels,
        cell_type_ids=cell_type_ids,
        train_pairs_file=args.train_pairs_file,
        val_pairs_file=args.val_pairs_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup(stage="fit")
    steps_per_epoch = len(dm.train_dataloader())
    total_steps = steps_per_epoch * args.epochs

    # Needed for pos_weight calculation
    dm.train_labels_npy = args.train_labels_npy

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    model = LitDNABindingModel(
        protein_in_dim=args.protein_in_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        use_cell_type=args.use_cell_type,
        num_cell_types=num_cell_types,
        cell_type_dim=args.cell_type_dim,
        fixed_tf_embs=fixed_tf_embs,
        fixed_tf_masks=fixed_tf_masks,
    )

    #model.set_global_pos_weight(dm)

    print("\n====== Model Summary ======")
    print(model)
    print("===========================\n")

    # -------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="{epoch:02d}-{val/pr_auc:.4f}-{val/loss:.4f}",
            monitor="val/pr_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/pr_auc",
            mode="max",
            patience=5,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    # -------------------------------------------------------
    # W&B logger
    # -------------------------------------------------------
    wandb_logger = None
    if args.wandb_project:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=args.output_dir,
            log_model=True,
        )

    # -------------------------------------------------------
    # Trainer
    # -------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=2000)] + callbacks,
        log_every_n_steps=2000,
        gradient_clip_val=1.0,
        deterministic=True,
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    # -------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------
    trainer.fit(model, datamodule=dm)
    print("\nTraining complete!\n")

    # -------------------------------------------------------


if __name__ == "__main__":
    main()




#seed42

'''
mkdir -p ....../results/seed42
mkdir -p ....../results/seed42/ckpts

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python train.py \
  --train_dna_npy ....../data/dna_data/train/train_data.npy \
  --train_labels_npy ....../data/dna_data/train/train_labels.npy \
  --train_metadata_tsv ....../data/metadata/seen_tf_metadata.tsv \
  --val_dna_npy ....../data/dna_data/val/val_data.npy \
  --val_labels_npy ....../data/dna_data/val/val_labels.npy \
  --val_metadata_tsv ....../data/metadata/seen_tf_metadata.tsv \
  --fixed_tf_embs_pt ....../data/tf_data/fixed_length_200/seen_tf/fixed_tf_embs.pt \
  --fixed_tf_masks_pt ....../data/tf_data/fixed_length_200/seen_tf/fixed_tf_masks.pt \
  --train_pairs_file ....../data/cached_pairs/seed42/seentf/train_pairs.npy \
  --val_pairs_file ....../data/cached_pairs/seed42/seentf/val_pairs.npy \
  --use_cell_type \
  --cell_type_dim 16 \
  --cell_type_ids_npy ....../data/metadata/seen_cell_type_ids.npy \
  --protein_in_dim 1024 \
  --epochs 20 \
  --batch_size 1024 \
  --num_workers 6 \
  --lr 1e-4 \
  --wandb_project tfbind \
  --run_name tfbind_seed42 \
  --output_dir ....../results/seed42/ckpts \
  > ....../results/seed42/tfbind_train_seed42.log 2>&1 &

'''

