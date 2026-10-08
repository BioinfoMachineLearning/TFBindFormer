import os
import re
import argparse
import torch
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn.functional as F


def embedding_features(seq_1d, seq_3di, tokenizer, model, device, debug=False):
    # clean sequences first
    seq_1d = str(seq_1d).strip().replace("*", "")
    seq_3di = str(seq_3di).strip().replace("*", "")

    seq_1d = re.sub(r"[UZOB]", "X", seq_1d)

    d1 = len(seq_1d)
    d2 = len(seq_3di)

    # preprocess sequences for ProstT5
    seq_1d_spaced = " ".join(list(seq_1d))
    seq_3di_spaced = " ".join(list(seq_3di.lower()))

    input_seqs = [
        "<AA2fold> " + seq_1d_spaced,
        "<fold2AA> " + seq_3di_spaced,
    ]

    ids = tokenizer(
        input_seqs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
        )

    emb_aa = outputs.last_hidden_state[0, 1 : d1 + 1].float()
    emb_3di = outputs.last_hidden_state[1, 1 : d2 + 1].float()

    # safer than using original d1, d2 only
    L = min(emb_aa.shape[0], emb_3di.shape[0])

    emb_aa = F.layer_norm(emb_aa[:L], emb_aa[:L].shape[-1:])
    emb_3di = F.layer_norm(emb_3di[:L], emb_3di[:L].shape[-1:])

    # mean fusion: L x 1024
    emb = (emb_aa + emb_3di) / 2

    if debug and L > 1:
        print("AA length:", d1)
        print("3Di length:", d2)
        print("emb_aa:", emb_aa.shape)
        print("emb_3di:", emb_3di.shape)
        print("final emb:", emb.shape)
        print("row std:", emb.std(dim=0).mean().item())
        print("pos0-pos1 max diff:", (emb[0] - emb[1]).abs().max().item())

    return emb.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract TF embeddings using ProstT5 AA + 3Di mean fusion"
    )
    parser.add_argument("--aa_dir", required=True, help="Directory with AA FASTA files")
    parser.add_argument("--di_fasta", required=True, help="Foldseek 3Di FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading ProstT5 tokenizer and model once...")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5",
        do_lower_case=False,
    )

    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5"
    ).to(device)

    model.eval()

    if device.type == "cpu":
        model.float()
    else:
        model.half()

    print("[INFO] Loading 3Di sequences...")
    di_dict = {
        rec.id.split()[0]: str(rec.seq)
        for rec in SeqIO.parse(args.di_fasta, "fasta")
    }

    for fname in sorted(os.listdir(args.aa_dir)):
        if not fname.endswith(".fasta"):
            continue

        tf_id = fname.replace(".fasta", "")
        aa_path = os.path.join(args.aa_dir, fname)

        if tf_id not in di_dict:
            print(f"⚠️ No 3Di for {tf_id}, skipping")
            continue

        aa_seq = str(next(SeqIO.parse(aa_path, "fasta")).seq)
        di_seq = di_dict[tf_id]

        emb = embedding_features(
            aa_seq,
            di_seq,
            tokenizer,
            model,
            device,
            debug=args.debug,
        )

        out_path = os.path.join(args.out_dir, f"{tf_id}_embedding.pt")
        torch.save(emb, out_path)

        print(f"Saved {tf_id}: {tuple(emb.shape)} → {out_path}")


if __name__ == "__main__":
    main()


"""

nohup python extract_tf_embeddings.py \
  --aa_dir /bml/ping/tfbind/data/1D_3Di/selected_fasta \
  --di_fasta /bml/ping/tfbind/data/1D_3Di/pdb_3Di_ss.fasta \
  --out_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings_1024_mean \
  --device cuda \
  > extract_tf_embeddings_1024_mean.log 2>&1 &


"""
#for debugging
'''
python extract_tf_embeddings.py \
  --aa_dir /bml/ping/tfbind/data/fasta \
  --di_fasta /bml/ping/tfbind/data/1D_3Di/pdb_3Di_ss.fasta \
  --out_dir /bmlfast/ping/tfbind_review/TFBindFormer/tf_embeddings_1024_mean \
  --device cuda \
  --debug
'''
