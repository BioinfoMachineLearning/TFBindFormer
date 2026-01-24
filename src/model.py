# ================================================================
# it_binding_model.py 
# ================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import wandb

from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryAccuracy,
        BinaryPrecision,
        BinaryRecall,
        BinaryF1Score,
    )

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

from src.architectures.binding_predictor import DNABindingPredictor


# ================================================================
# STEP-BASED WARMUP + COSINE LR
# ================================================================

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        # warmup
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # cosine decay
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cos_scale = 0.5 * (1 + np.cos(np.pi * progress))
        return [base_lr * cos_scale for base_lr in self.base_lrs]



# ================================================================
#  LIT MODEL — BALANCED TRAINING, STEP LR, THRESHOLD SWEEP
# ================================================================

class LitDNABindingModel(pl.LightningModule):

    def __init__(
        self,
        protein_in_dim=512,
        d_model=128,
        dropout=0.3,
        lr=1e-4,
        weight_decay=1e-5,
        warmup_steps=1000,     # STEP-BASED WARMUP
        total_steps=None,      # COMPUTED AUTOMATICALLY IF None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DNABindingPredictor(
            protein_in_dim=protein_in_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps  # If None → infer automatically

        # Balanced sampling 
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_logits = []
        self.val_targets = []

        self.test_probs = []
        self.test_targets = []

        self.best_threshold = None


    # ----------------------------------------------------------
    def _pad_proteins(self, emb_list, device):
        cleaned = [t.squeeze(0) if t.ndim == 3 else t for t in emb_list]
        B = len(cleaned)
        Lmax = max(t.shape[0] for t in cleaned)
        D = cleaned[0].shape[1]
        out = torch.zeros((B, Lmax, D), device=device)
        mask = torch.ones((B, Lmax), dtype=torch.bool, device=device)
        for i, t in enumerate(cleaned):
            L = t.shape[0]
            out[i, :L] = t
            mask[i, :L] = False
        return out, mask


    # ----------------------------------------------------------
    # OPTIMIZER + STEP LR SCHEDULER
    # ----------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        sched = WarmupCosineLR(
            opt,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
    }



    # ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        dna, tf_embs, labels, tf_idx = batch
        prot, mask = self._pad_proteins(tf_embs, dna.device)
        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)
        loss = self.loss_fn(logits, labels.float())
        self.log("train/loss", loss, on_epoch=True, batch_size=dna.size(0))
        return loss


    # ----------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        dna, tf_embs, labels, tf_idx = batch
        prot, mask = self._pad_proteins(tf_embs, dna.device)
        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)

        loss = self.loss_fn(logits, labels.float())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=dna.size(0))

        self.val_logits.append(torch.sigmoid(logits).detach().cpu())
        self.val_targets.append(labels.detach().cpu())
        return loss


    # ----------------------------------------------------------
    # VALIDATION — threshold sweep + PR/ROC logging
    # ----------------------------------------------------------
    def on_validation_epoch_end(self):
        if not self.val_logits:
            return

        probs = torch.cat(self.val_logits, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1).int()

        self.val_logits.clear()
        self.val_targets.clear()

        # AUROC / AUPRC
        auroc = BinaryAUROC()(probs, targets)
        auprc = BinaryAveragePrecision()(probs, targets)
        self.log("val/roc_auc", auroc, prog_bar=True)
        self.log("val/pr_auc", auprc, prog_bar=True)

        # Threshold sweep
        thresholds = torch.linspace(0.0, 0.3, steps=301)
        eps = 1e-8
        best_f1 = 0.0
        best_t = 0.0
        best_p = 0.0
        best_r = 0.0

        for t in thresholds:
            preds = (probs >= t).int()
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_p = float(precision)
                best_r = float(recall)

        self.best_threshold = best_t
        self.log("val/best_f1", best_f1, prog_bar=True)
        self.log("val/best_precision", best_p)
        self.log("val/best_recall", best_r)
        self.log("val/best_threshold", best_t)

        # PR / ROC curves to W&B
        try:
            p_np = probs.numpy()
            y_np = targets.numpy()

            pre, rec, _ = precision_recall_curve(y_np, p_np)
            fpr, tpr, _ = roc_curve(y_np, p_np)

            self.logger.experiment.log({
                "val/pr_curve": wandb.plot.line_series([rec], [pre], keys=["precision"], xname="Recall"),
                "val/roc_curve": wandb.plot.line_series([fpr], [tpr], keys=["TPR"], xname="FPR"),
            })
        except Exception as e:
            print("[WARN] PR/ROC curve error:", e)



# TEST
# ----------------------------------------------------------
    def test_step(self, batch, batch_idx):
        dna, tf_embs, labels, tf_idx = batch

        # pad proteins
        prot, mask = self._pad_proteins(tf_embs, dna.device)

        # forward
        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)

        # probabilities
        probs = torch.sigmoid(logits)

        # store
        self.test_probs.append(probs.detach().cpu())
        self.test_targets.append(labels.detach().cpu())

        # loss
        loss = self.loss_fn(logits, labels.float())

        # log
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    
    
    #macro metrices
    def on_test_epoch_end(self):
        

        parent_dir = "eval_out"
        data_dir = os.path.join(parent_dir, "metrics_data")
        out_dir  = os.path.join(parent_dir, "eval_results")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------
        # Collect predictions
        # --------------------------------------------------
        probs = torch.cat(self.test_probs, dim=0).view(-1)
        targets = torch.cat(self.test_targets, dim=0).view(-1).int()

        # to numpy (REQUIRED for sklearn)
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # --------------------------------------------------
        # Save raw data
        # --------------------------------------------------
        np.save(f"{data_dir}/probs.npy", probs_np)
        np.save(f"{data_dir}/targets.npy", targets_np)

        # --------------------------------------------------
        # Thresholded predictions
        # --------------------------------------------------
        thr = 0.3
        preds_np = (probs_np >= thr).astype(int)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        auroc = roc_auc_score(targets_np, probs_np)
        aupr  = average_precision_score(targets_np, probs_np)

        acc  = accuracy_score(targets_np, preds_np)
        prec = precision_score(targets_np, preds_np, zero_division=0)
        rec  = recall_score(targets_np, preds_np, zero_division=0)
        f1   = f1_score(targets_np, preds_np, zero_division=0)

        print("========== Test Metrics ==========")
        print(f"AUROC : {auroc:.4f}")
        print(f"AUPR  : {aupr:.4f}")
        print(f"ACC   : {acc:.4f}")
        print(f"Prec  : {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1    : {f1:.4f}")

        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(f"AUROC  {auroc:.6f}\n")
            f.write(f"AUPR   {aupr:.6f}\n")
            f.write(f"ACC    {acc:.6f}\n")
            f.write(f"PREC   {prec:.6f}\n")
            f.write(f"RECALL {rec:.6f}\n")
            f.write(f"F1     {f1:.6f}\n")

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------
        fpr, tpr, _ = roc_curve(targets_np, probs_np)
        np.savez(f"{out_dir}/roc_data.npz", fpr=fpr, tpr=tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auroc:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/roc_curve.png", dpi=300)
        plt.close()

        # --------------------------------------------------
        # Precision–Recall Curve
        # --------------------------------------------------
        precision, recall, _ = precision_recall_curve(targets_np, probs_np)
        np.savez(f"{out_dir}/pr_data.npz", precision=precision, recall=recall)

        plt.figure()
        plt.plot(recall, precision, label=f"PR (AUPRC={aupr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_curve.png", dpi=300)
        plt.close()

        print(f"\nAll results saved to: {out_dir}")




            