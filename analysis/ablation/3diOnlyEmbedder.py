import os
import torch
from Bio import SeqIO
from prostt5_embedder import ProstT5Embedder


# ---------------- paths ----------------
THREEDI_FASTA = ".../TFBindFormer/data/tf_data/3di_out/pdb_3Di_ss.fasta"
OUT_DIR = ".../TFBindFormer/data/tf_data/ablation/3di_embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- load embedder ----------------
embedder = ProstT5Embedder()
embedder.eval()

# ---------------- iterate over 3Di FASTA ----------------
for record in SeqIO.parse(THREEDI_FASTA, "fasta"):
    base_id = record.id.split()[0]
    seq_3di = str(record.seq)

    # ---- compute embedding ----
    with torch.no_grad():
        feat = embedder(
            seq_3di=seq_3di,
            mode="3di"
        )   # [L, 512]

    # ---- save tensor only ----
    out_path = os.path.join(
        OUT_DIR,
        f"{base_id}_L{feat.shape[0]}_512.pt"
    )

    torch.save(feat, out_path)

    print(f"✅ Saved 3Di-only tensor: {base_id} → {out_path}")



"""
nohup python 3diOnlyFeature.py > ouput_3diEmbedder.log 2>&1 &
"""