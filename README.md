# TFBindFormer

**TFBindFormer** is a hybrid cross-attention Transformer model for
**transcription factor (TF)–DNA binding prediction**. The model
explicitly integrates transcription factor protein representations
derived from **amino-acid sequence** and **protein structural context**
with **genomic DNA sequence bins**, enabling position-specific
modeling of protein–DNA interactions beyond sequence-only approaches.

TFBindFormer is designed for **genome-wide TF binding prediction under
severe class imbalance** and demonstrates improved ranking and
enrichment of bona fide binding sites compared with representative
state-of-the-art models.

---

## Features

- Hybrid cross-attention architecture for explicit residue–nucleotide interactions  
- Integration of TF amino-acid sequence and protein structure information  
- Genome-wide TF binding prediction under extreme class imbalance  
- Modular design for ablation and extension  
- Reproducible training and evaluation pipeline  

---

## Repository Structure

```text
TFBindFormer/
├── data/
│   ├── dna_data/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── tf_data/
│       ├── tf_aa_sequence/
│       ├── tf_structure/
│       └── metadata_tfbs.tsv
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── extract_tf_embeddings.py
│   └── generate_3di_tokens.sh
│
├── src/
│   ├── architectures/
│   │   ├── binding_predictor.py
│   │   ├── cross_attention_encoder.py
│   │   └── tbinet_dna_encoder.py
│   │
│   ├── model.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

- **data/**: DNA sequence data, TF protein data, and metadata  
- **scripts/**: Training, evaluation, and preprocessing scripts  
- **src/architectures/**: Core model components and attention modules  
- **src/model.py**: TFBindFormer model wrapper  
- **src/utils.py**: Shared utilities and helper functions

---

## Quick Start

### 1. Create environment and install dependencies

```bash
conda create -n tfbindformer python=3.9
conda activate tfbindformer
pip install -r requirements.txt

```
---

### 2. Download dataset

All DNA and transcription factor (TF) data used in this project are  
publicly available on **Zenodo**.

Please download the dataset and place it under the `data/` directory  
following the structure described above.

---

### 3. Generate 3Di structural tokens 

```bash
chmod +x scripts/generate_3di_tokens.sh
scripts/generate_3di_tokens.sh <pdb_dir> <output_dir>
```

### 4. Generate TF protein embeddings

```bash
nohup python scripts/extract_tf_embeddings.py \
  --aa_dir data/tf_data/tf_aa_sequence \
  --di_fasta data/tf_data/3di_out/pdb_3Di_ss.fasta \
  --out_dir data/tf_data/tf_embeddings \
  > extract_tf_embeddings.log 2>&1 &
```

### 5. Train TFBindFormer

```bash
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
```

### 6. Evaluation

```bash
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
```
---
## Citation

If you use TFBindFormer in your work, please cite the associated
manuscript. Citation details will be updated upon publication.

---

## Contact

For questions or issues, please open an issue in this repository or
contact the authors.

---

