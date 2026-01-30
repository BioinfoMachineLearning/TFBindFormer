import os
import torch
from Bio import SeqIO
from prostt5_embedder import ProstT5Embedder


AA_FASTA_DIR = ".../TFBindFormer/data/tf_data/tf_sequence"
OUT_DIR = ".../TFBindFormer/data/tf_data/ablation/aa_embeddings"
MODE = "aa"

os.makedirs(OUT_DIR, exist_ok=True)

embedder = ProstT5Embedder()
embedder.eval()

for fname in os.listdir(AA_FASTA_DIR):
    if not fname.endswith(".fasta"):
        continue

    base_id = fname.replace(".fasta", "")
    aa_path = os.path.join(AA_FASTA_DIR, fname)
    aa_seq = str(next(SeqIO.parse(aa_path, "fasta")).seq)

    with torch.no_grad():
        feat = embedder(seq_1d=aa_seq, mode="aa")  # [L,512]

    out_path = os.path.join(
        OUT_DIR, f"{base_id}_L{feat.shape[0]}_512.pt"
    )

    torch.save(feat, out_path)

    print(f"✅ Saved AA-only tensor: {base_id} → {out_path}")



"""
nohup python aaOnlyFeature.py > ouput_aaEmbedder.log 2>&1 &

"""