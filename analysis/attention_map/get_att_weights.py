import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import torch

#load model
from lit_binding_model import LitDNABindingModel # adjust import

import matplotlib.pyplot as plt
import seaborn as sns




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test DNA (one-hot)
test_dna = np.load(".../TFBindFormer/data/dna_data/test/test_oneHot.npy")        # (455024, 1000, 4), uint8
test_dna = test_dna.astype(np.float32)     # model expects float

# Load test labels
test_labels = np.load(".../TFBindFormer/data/dna_data/test/test_labels.npy")   # (455024, num_TFs)

print(test_dna.shape, test_labels.shape)

ctcf_j = 2

labels_ctcf = test_labels[:, ctcf_j]

pos_idx = np.where(labels_ctcf == 1)[0]
neg_idx = np.where(labels_ctcf == 0)[0]

np.random.seed(42)
pos_sel = np.random.choice(pos_idx, 5, replace=False)
neg_sel = np.random.choice(neg_idx, 5, replace=False)

selected_idx = np.concatenate([pos_sel, neg_sel])
selected_labels = labels_ctcf[selected_idx]

print(selected_labels)   # first 5 → 1, last 5 → 0

dna_sel = test_dna[selected_idx]     # (10, 1000, 4)
dna_sel = torch.tensor(dna_sel).to(device)


#tf embedding
ctcf_emb_path = ".../TFBindFormer/data/tf_data/tf_embeddings/CTCF_P49711_embedding.pt"

ctcf_emb = torch.load(ctcf_emb_path, weights_only=True)

# remove batch dim if present
if ctcf_emb.ndim == 3:
    ctcf_emb = ctcf_emb.squeeze(0)

ctcf_emb = ctcf_emb.to(device)

print(ctcf_emb.shape)

B = dna_sel.shape[0]  # should be 10

protein_batch = ctcf_emb.unsqueeze(0).repeat(B, 1, 1)



ckpt_path = ".../checkpoints/..../epoch=18-val/roc_auc=0.9587-val/loss=0.2019.ckpt"    #choose best model from checkpoints

model = LitDNABindingModel.load_from_checkpoint(
    ckpt_path,
    map_location=device
)

model = model.to(device)


model.eval()
with torch.no_grad():
    preds, attn = model(
        dna_onehot=dna_sel,
        protein_emb=protein_batch,
        return_attention=True
    )

print(preds.shape)
print("type(attn):", type(attn))
print("keys:", attn.keys())

print("num DNA→TF layers:", len(attn["dna_to_prot"]))
print("DNA→TF last layer shape:", attn["dna_to_prot"][-1].shape)

if len(attn["prot_to_dna"]) > 0:
    print("TF→DNA last layer shape:", attn["prot_to_dna"][-1].shape)

#save preds,label,dna index information

out_dir = "heatmap/outputs"
attn_dir = os.path.join(out_dir, "attn_weights")
fig_dir  = os.path.join(out_dir, "figures")

os.makedirs(attn_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ----------------------------
# Save combined predictions as TXT
# ----------------------------
preds_np = torch.sigmoid(preds).detach().cpu().numpy().reshape(-1) # (10,)
labels_np = selected_labels.astype(int)                # (10,)
idx_np = selected_idx                                  # (10,)

pred_dir = os.path.join(out_dir, "predictions_txt")
os.makedirs(pred_dir, exist_ok=True)

out_txt = os.path.join(pred_dir, "preds_labels_windows.txt")

combined = np.column_stack([idx_np, labels_np, preds_np])

np.savetxt(
    out_txt,
    combined,
    fmt=["%d", "%d", "%.6f"],
    delimiter="\t",
    header="window_idx\tlabel\tprediction",
    comments=""
)

print(f"Saved combined TXT: {out_txt}")






#save weight to local
# ---- DNA → TF (last layer) ----
dna_tf = attn["dna_to_prot"][-1]     # (B, heads, 200, 200)
dna_tf = dna_tf.mean(dim=1)          # (B, 200, 200)
dna_tf = dna_tf.cpu().numpy()

# ---- TF → DNA (last layer) ----
tf_dna = attn["prot_to_dna"][-1]     # (B, heads, 200, 200)
tf_dna = tf_dna.mean(dim=1)          # (B, 200, 200)
tf_dna = tf_dna.cpu().numpy()

# Save
np.save(os.path.join(attn_dir, "dna_to_tf_attn.npy"), dna_tf)
np.save(os.path.join(attn_dir, "tf_to_dna_attn.npy"), tf_dna)

print("Saved attention weights:")
print(" ", os.path.join(attn_dir, "dna_to_tf_attn.npy"))
print(" ", os.path.join(attn_dir, "tf_to_dna_attn.npy"))


#plot heatmap

#DNA → TF (core on DNA only), dna core region 80-120 
def plot_dna_to_tf(mat, title, fname):
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, cmap="viridis", xticklabels=False, yticklabels=False)

    # DNA core (rows)
    plt.axhline(80, color="red", linestyle="--")
    plt.axhline(120, color="red", linestyle="--")

    plt.ylabel("DNA positions (200)")
    plt.xlabel("TF residues (200)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

#TF → DNA (core on DNA only)
def plot_tf_to_dna(mat, title, fname):
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, cmap="viridis", xticklabels=False, yticklabels=False)

    # DNA core (columns)
    plt.axvline(80, color="red", linestyle="--")
    plt.axvline(120, color="red", linestyle="--")

    plt.xlabel("DNA positions (200)")
    plt.ylabel("TF residues (200)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()




window_ids = range(len(selected_labels))
#dna->tf
for i in window_ids:
    label = "pos" if selected_labels[i] == 1 else "neg"
    fname = os.path.join(fig_dir, f"dna_to_tf_{label}_win{i}_core.png")

    plot_dna_to_tf(
        dna_tf[i],
        title=f"CTCF {label.upper()} : DNA → TF (window {i})",
        fname=fname,
    )

#tf->dna
for i in window_ids:
    label = "pos" if selected_labels[i] == 1 else "neg"
    fname = os.path.join(fig_dir, f"tf_to_dna_{label}_win{i}.png")

    plot_tf_to_dna(
        tf_dna[i],
        title=f"CTCF {label.upper()} : TF → DNA (window {i})",
        fname=fname,
    )


#nohup python heatmap/heatmap.py > heatmap/heatmap.log 2>&1 &
