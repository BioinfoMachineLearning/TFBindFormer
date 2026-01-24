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
# 1. TF ALIAS MAPPING (ONLY these should be canonicalized)
# ============================================================

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
    """Return canonical TF name ONLY if present in alias map."""
    return TF_ALIAS_MAP.get(tf, tf)


# ============================================================
# 2. EMBEDDING LOADING WITH ORDER MAPPING
# ============================================================

def load_embedding_index(embedding_dir):
    """
    Build mapping: canonical_name → embedding_file_path
    Handles filenames like:
        c-Fos_P01100_embedding.pt
        AP-2alpha_P05549_embedding.pt
        REST_Q12345_embedding.pt
    """
    emb_index = {}

    for fname in os.listdir(embedding_dir):
        if not fname.endswith(".pt"):
            continue

        # strip suffix
        core = fname.replace("_embedding.pt", "")   # e.g. "c-Fos_P01100"

        # split at first underscore
        tf_raw = core.split("_")[0]                 # -> "c-Fos"

        # convert to canonical
        tf_canon = canonical_name(tf_raw)           # alias mapping applied only if needed

        emb_index[tf_canon] = os.path.join(embedding_dir, fname)

    return emb_index


def load_tf_embeddings_in_label_order(tf_names, embedding_dir):
    """
    Load embeddings in same column order as label matrix.
    """
    emb_index = load_embedding_index(embedding_dir)

    tf_embs = []
    canon_names = []
    missing = []

    for tf in tf_names:
        canon = canonical_name(tf)
        canon_names.append(canon)

        if canon not in emb_index:
            missing.append((tf, canon))
            tf_embs.append(None)
            continue

        emb = torch.load(emb_index[canon], weights_only=True).float()
        if emb.ndim == 3:
            emb = emb.squeeze(0)

        tf_embs.append(emb)

    if missing:
        print("\n[WARNING] Missing TF embeddings:")
        for orig, canon in missing:
            print(f"  Label TF '{orig}' → Canonical '{canon}' has no embedding file")

    return tf_embs, canon_names



# ============================================================
# 3. POSITIVE + NEGATIVE DOWNSAMPLING
# ============================================================

def build_sample_indices(labels, neg_fraction=0.001, seed=2024):
    """
    Build list of (dna_idx, tf_idx) pairs:

      - include ALL positives
      - include a random subset of negatives
        (each negative is kept with probability = neg_fraction)

    This avoids ever storing all negatives in memory.
    """
    rng = np.random.default_rng(seed)
    N, T = labels.shape

    pos = []
    neg = []

    for i in range(N):
        row = labels[i]

        # ---- positives for this DNA window ----
        pos_js = np.where(row == 1)[0]
        for j in pos_js:
            pos.append((i, j))

        # ---- sampled negatives for this DNA window ----
        if neg_fraction > 0.0:
            # candidates where label == 0
            neg_js = np.where(row == 0)[0]
            if len(neg_js) > 0:
                # Bernoulli sampling per negative
                keep_mask = rng.random(len(neg_js)) < neg_fraction
                for j in neg_js[keep_mask]:
                    neg.append((i, j))

    pos = np.asarray(pos, dtype=np.int32)
    neg = np.asarray(neg, dtype=np.int32)

    all_pairs = np.concatenate([pos, neg], axis=0)
    rng.shuffle(all_pairs)

    print(f"[Sampling] Pos={len(pos)}, Neg={len(neg)}, Total={len(all_pairs)}")

    return all_pairs


# ============================================================
# 4. DATASET FOR TF–DNA COMBINATIONS
# ============================================================

class TFTargetDataset(Dataset):
    """DNA + TF embedding + label"""

    def __init__(self, dna_data, tf_embs, labels, sample_indices):
        self.dna_data = dna_data       # numpy array (N,1000,4)
        self.tf_embs = tf_embs         # list of embeddings
        self.labels = labels           # numpy array (N,T)
        self.samples = sample_indices  # array of (dna_idx, tf_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dna_i, tf_j = self.samples[idx]

        dna = torch.tensor(self.dna_data[dna_i], dtype=torch.float32)
        tf_emb = self.tf_embs[tf_j]  # tensor (L,512)
        label = float(self.labels[dna_i, tf_j])

        return dna, tf_emb, label, tf_j


# ============================================================
# 5. COLLATE FUNCTION
# ============================================================

def tfbind_collate(batch):
    dna_batch = torch.stack([x[0] for x in batch])   # (B,1000,4)
    tf_emb_list = [x[1] for x in batch]              # list[B] of (L,512)
    labels = torch.tensor([x[2] for x in batch]).float()
    tf_idx = torch.tensor([x[3] for x in batch]).long()
    return dna_batch, tf_emb_list, labels, tf_idx


# ============================================================
# 6. PYTORCH LIGHTNING DATAMODULE
# ============================================================


class TFBindDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dna, train_labels,
        val_dna, val_labels,
        test_dna=None, test_labels=None,   # <-- now optional
        tf_embs=None,
        batch_size=128,
        neg_fraction=0.01,
        num_workers=4,
        #Slimit_examples=None,
    ):
        super().__init__()

        # Required splits
        self.train_dna = train_dna
        self.train_labels = train_labels

        self.val_dna = val_dna
        self.val_labels = val_labels

        # Optional test split
        self.test_dna = test_dna
        self.test_labels = test_labels

        self.tf_embs = tf_embs

        self.batch_size = batch_size
        self.neg_fraction = neg_fraction
        self.num_workers = num_workers
        #self.limit_examples = limit_examples

    
    # -------------------------------------------------------
    
    def setup(self, stage=None):

        # ----------------------------
        # TRAINING (fit)
        # ----------------------------
        if stage in ("fit", None):
            # ---- Train ----
            train_pairs = build_sample_indices(
                self.train_labels,
                neg_fraction=self.neg_fraction
            )
            self.train_dataset = TFTargetDataset(
                self.train_dna, self.tf_embs,
                self.train_labels, train_pairs
            )

            # ---- Validation ----
            val_pairs = build_sample_indices(
                self.val_labels,
                neg_fraction=0.08
            )
            self.val_dataset = TFTargetDataset(
                self.val_dna, self.tf_embs,
                self.val_labels, val_pairs
            )

        # ----------------------------
        # TESTING (test)
        # ----------------------------
        if stage in ("test", None):
            if self.test_dna is not None and self.test_labels is not None:
                print("[INFO] Building TEST dataset (neg_fraction=1.0)")

                test_pairs = build_sample_indices(
                    self.test_labels,
                    neg_fraction=1.0      # sample all positives + all negatives
                )
                self.test_dataset = TFTargetDataset(
                    self.test_dna, self.tf_embs,
                    self.test_labels, test_pairs
                )
            else:
                print("[INFO] No test dataset provided.")
                self.test_dataset = None

    

    # -------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=tfbind_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=tfbind_collate,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=tfbind_collate,
        )


    '''
    #test train on 20 examples
    def setup(self, stage=None):

        # ---------------------------------------------------
        # Optional: Limit number of examples for debugging
        # ---------------------------------------------------
        if self.limit_examples is not None:
            n = self.limit_examples
            print(f"[DEBUG] Limiting train/val/test to first {n} examples (if available).")

            # train always exists
            self.train_dna = self.train_dna[:n]
            self.train_labels = self.train_labels[:n]

            # val always exists
            self.val_dna = self.val_dna[:n]
            self.val_labels = self.val_labels[:n]

            # test may be None → check before slicing
            if self.test_dna is not None:
                self.test_dna = self.test_dna[:n]
                self.test_labels = self.test_labels[:n]
            else:
                print("[DEBUG] No test set loaded (skipping test slicing).")

        # ---------------------------------------------------
        # Build (dna_idx, tf_idx) pairs
        # ---------------------------------------------------
        train_pairs = build_sample_indices(self.train_labels, neg_fraction=self.neg_fraction)
        val_pairs   = build_sample_indices(self.val_labels,   neg_fraction=0.5)

        # test_labels may be None → test_pairs should be empty
        if self.test_labels is not None:
            test_pairs  = build_sample_indices(self.test_labels, neg_fraction=1.0)
        else:
            test_pairs = []

        # ---------------------------------------------------
        # Build datasets
        # ---------------------------------------------------
        self.train_dataset = TFTargetDataset(self.train_dna, self.tf_embs, self.train_labels, train_pairs)
        self.val_dataset   = TFTargetDataset(self.val_dna,   self.tf_embs, self.val_labels,   val_pairs)

        if self.test_dna is not None:
            self.test_dataset = TFTargetDataset(self.test_dna, self.tf_embs, self.test_labels, test_pairs)
        else:
            self.test_dataset = None

    
'''   