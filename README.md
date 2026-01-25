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
