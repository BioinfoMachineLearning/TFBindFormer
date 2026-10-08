# tfbind_pipeline.py
# ============================================================
#  COMPLETE TF-BINDING DATA PIPELINE 
#  Supports:
#       - TF alias mapping (only for specified TFs)
#       - Canonicalized embedding loading
#       - Exact alignment with label columns
#       - Positive + downsampled negative sampling
#       - Stable PyTorch Lightning DataModule
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ============================================================
# DATASET FOR TF–DNA COMBINATIONS
# ============================================================

class TFTargetDataset(Dataset):
    """
    Dataset for cached TF-DNA pairs.

    Returns:
        dna:          (1000, 4)
        label:        scalar
        tf_idx:       task/label column index
        cell_type_id: task-level cell type ID
    """

    def __init__(
        self,
        dna_data,
        labels,
        sample_indices,
        cell_type_ids=None,
    ):
        self.dna_data = dna_data
        self.labels = labels
        self.samples = sample_indices
        self.cell_type_ids = cell_type_ids

        assert self.samples.ndim == 2 and self.samples.shape[1] == 2, (
            f"sample_indices should have shape (N, 2), got {self.samples.shape}"
        )

        if self.cell_type_ids is not None:
            assert labels.shape[1] == len(self.cell_type_ids), (
                f"labels has {labels.shape[1]} tasks, "
                f"but cell_type_ids has {len(self.cell_type_ids)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dna_i, tf_j = self.samples[idx]

        dna = torch.tensor(self.dna_data[dna_i], dtype=torch.float32)
        label = float(self.labels[dna_i, tf_j])

        if self.cell_type_ids is None:
            cell_type_id = -1
        else:
            cell_type_id = int(self.cell_type_ids[tf_j])

        return dna, label, int(tf_j), cell_type_id


# ============================================================
# 5. COLLATE FUNCTION
# ============================================================

def tfbind_collate(batch):
    dna_batch = torch.stack([x[0] for x in batch])       # (B,1000,4)
    labels = torch.tensor([x[1] for x in batch]).float()
    tf_idx = torch.tensor([x[2] for x in batch]).long()
    cell_type_ids = torch.tensor([x[3] for x in batch]).long()

    return dna_batch, labels, tf_idx, cell_type_ids


# ============================================================
# 6. PYTORCH LIGHTNING DATAMODULE
# ============================================================

class TFBindDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dna=None,
        train_labels=None,
        val_dna=None,
        val_labels=None,
        test_dna=None,
        test_labels=None,
        cell_type_ids=None,
        train_pairs_file=None,
        val_pairs_file=None,
        test_pairs_file=None,
        batch_size=128,
        num_workers=4,
    ):
        super().__init__()

        self.train_dna = train_dna
        self.train_labels = train_labels

        self.val_dna = val_dna
        self.val_labels = val_labels

        self.test_dna = test_dna
        self.test_labels = test_labels

        self.cell_type_ids = cell_type_ids

        self.train_pairs_file = train_pairs_file
        self.val_pairs_file = val_pairs_file
        self.test_pairs_file = test_pairs_file

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        if stage in ("fit", None):
            if self.train_pairs_file is None or self.val_pairs_file is None:
                raise ValueError("train_pairs_file and val_pairs_file are required for training.")

            print("[INFO] Loading cached train pairs:", self.train_pairs_file)
            train_pairs = np.load(self.train_pairs_file)

            print("[INFO] Loading cached val pairs:", self.val_pairs_file)
            val_pairs = np.load(self.val_pairs_file)

            print("[INFO] train_pairs:", train_pairs.shape)
            print("[INFO] val_pairs:", val_pairs.shape)

            self.train_dataset = TFTargetDataset(
                dna_data=self.train_dna,
                labels=self.train_labels,
                sample_indices=train_pairs,
                cell_type_ids=self.cell_type_ids,
            )

            self.val_dataset = TFTargetDataset(
                dna_data=self.val_dna,
                labels=self.val_labels,
                sample_indices=val_pairs,
                cell_type_ids=self.cell_type_ids,
            )

        if stage in ("test", None):
            if self.test_dna is not None and self.test_labels is not None:
                if self.test_pairs_file is None:
                    raise ValueError("test_pairs_file is required for testing.")

                print("[INFO] Loading cached test pairs:", self.test_pairs_file)
                test_pairs = np.load(self.test_pairs_file)
                print("[INFO] test_pairs:", test_pairs.shape)

                self.test_dataset = TFTargetDataset(
                    dna_data=self.test_dna,
                    labels=self.test_labels,
                    sample_indices=test_pairs,
                    cell_type_ids=self.cell_type_ids,
                )
            else:
                print("[INFO] No test dataset provided.")
                self.test_dataset = None

    def _loader_kwargs(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": tfbind_collate,
            "pin_memory": torch.cuda.is_available(),
        }

        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2

        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    