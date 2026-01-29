#!/usr/bin/env python3
"""
Test TF–DNA Binding Predictor
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

# append project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import TFBindDataModule, load_tf_embeddings_in_label_order
from src.model import LitDNABindingModel


############################################
# ARGUMENT PARSER
############################################
def parse_args():
    parser = argparse.ArgumentParser(description="Test GLOBAL TF-DNA Binding Predictor")

    parser.add_argument("--test_dna_npy", type=str, required=True)
    parser.add_argument("--test_labels_npy", type=str, required=True)
    parser.add_argument("--test_metadata_tsv", type=str, required=True)

    parser.add_argument("--embedding_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="tfbind-global-test")

    return parser.parse_args()


############################################
# MAIN
############################################
def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    print("[INFO] Loading test dataset…")
    test_dna = np.load(args.test_dna_npy, mmap_mode="r")
    test_labels = np.load(args.test_labels_npy, mmap_mode="r")

    print("[INFO] Loading metadata…")
    meta = pd.read_csv(args.test_metadata_tsv, sep="\t")
    tf_names = meta["TF/DNase/HistoneMark"].tolist()

    '''
    #test on 20 examples
    LIMIT = 40
    print(f"[INFO] Limiting test set to first {LIMIT} examples…")

    test_dna = test_dna[:LIMIT]
    test_labels = test_labels[:LIMIT]
    '''

    print("[INFO] Loading TF embeddings…")
    tf_embs, canon_names = load_tf_embeddings_in_label_order(tf_names, args.embedding_dir)

    print("[INFO] Preparing DataModule…")
    dm = TFBindDataModule(
        train_dna=None, train_labels=None,
        val_dna=None, val_labels=None,
        test_dna=test_dna,
        test_labels=test_labels,
        tf_embs=tf_embs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_fraction=0.0,  # irrelevant for test
    )

    dm.setup(stage="test")

    # ----------------------------
    # Load model checkpoint
    # ----------------------------
    print(f"[INFO] Loading model from: {args.ckpt_path}")
    model = LitDNABindingModel.load_from_checkpoint(
    args.ckpt_path,
    weights_only=False
    )
    model.eval()   # ensures dropout OFF

    # ----------------------------
    # W&B Logger
    # ----------------------------
    wandb_logger = None
    if args.wandb_project:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            log_model=False,
        )

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=500)],
    )

    # ----------------------------
    # RUN TEST
    # ----------------------------
    print("\n==============================")
    print(" Running TEST evaluation…")
    print("==============================\n")

    trainer.test(model=model, datamodule=dm, ckpt_path=None)

    print("\n==============================")
    print(" Test Complete!")
    print("==============================\n")




if __name__ == "__main__":
    main()







'''
nohup python eval.py \
  --test_dna_npy ../TFBindFormer/data/dna_data/test/test_oneHot.npy \
  --test_labels_npy ../TFBindFormer/data/dna_data/test/test_labels.npy \
  --test_metadata_tsv ../TFBindFormer/data/tf_data/metadata_tfbs.tsv \
  --embedding_dir ../TFBindFormer/data/tf_data/tf_embeddings_512 \
  --ckpt_path ../checkpoints/---.ckpt \
  --batch_size 1024 \
  --wandb_project tfbind_eval \
  --run_name tfbind_eval \
  > tfbind_eval.log 2>&1 &
  '''
