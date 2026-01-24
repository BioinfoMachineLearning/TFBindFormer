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

from src.utils import (
    TFBindDataModule,
    load_tf_embeddings_in_label_order,
)
from src.model import LitDNABindingModel


########################################
# Deterministic Behavior
########################################
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
pl.seed_everything(SEED, workers=True)

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

    parser.add_argument("--embedding_dir", type=str, required=True)

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
    print("[INFO] Loading metadata...")
    meta = pd.read_csv(args.train_metadata_tsv, sep="\t")
    tf_names = meta["TF/DNase/HistoneMark"].tolist()

    print("[INFO] Loading TF embeddings...")
    tf_embs, canon_names = load_tf_embeddings_in_label_order(tf_names, args.embedding_dir)

    # -------------------------------------------------------
    # DataModule
    # -------------------------------------------------------
    dm = TFBindDataModule(
        train_dna=train_dna,
        train_labels=train_labels,
        val_dna=val_dna,
        val_labels=val_labels,
        #test_dna=test_dna,
        #test_labels=test_labels,
        tf_embs=tf_embs,
        batch_size=args.batch_size,
        neg_fraction=args.neg_fraction,
        num_workers=args.num_workers,
        #limit_examples=20,
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,     
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
            filename="{epoch:02d}-{val/roc_auc:.4f}-{val/loss:.4f}",
            monitor="val/roc_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
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





'''
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python train.py \
  --train_dna_npy ../TFBindFormer/data/dna_data/train/train_oneHot.npy \
  --train_labels_npy ../TFBindFormer/data/dna_data/train/train_labels.npy \
  --train_metadata_tsv ../TFBindFormer/data/tf_data/metadata_tfbs.tsv \
  --val_dna_npy ../TFBindFormer/data/dna_data/val/valid_oneHot.npy \
  --val_labels_npy ../TFBindFormer/data/dna_data/val/valid_labels.npy \
  --val_metadata_tsv /bml/ping/TFBindFormer/data/tf_data/metadata_tfbs.tsv \
  --embedding_dir ../TFBindFormer/data/tf_data/tf_embeddings \
  --epochs 20 \
  --batch_size 1024 \
  --num_workers 6 \
  --lr 1e-4 \
  --neg_fraction 0.015 \
  --wandb_project tfbind-train \
  --run_name tfbind_train \
  --output_dir ./checkpoints/tfbind_train \
  > tfbind_train.log 2>&1 &

'''


