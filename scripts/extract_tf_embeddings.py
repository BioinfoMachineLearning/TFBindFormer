import os
import re
import argparse
import torch
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel


def embedding_features(seq_1d, seq_3di, device):
    d1 = len(seq_1d)
    d2 = len(seq_3di)

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5"
    ).to(device)

    if device.type == "cpu":
        model.float()
    else:
        model.half()

    # preprocess sequences
    seq_1d = " ".join(list(re.sub(r"[UZOB]", "X", seq_1d)))
    seq_3di = " ".join(list(seq_3di.lower()))

    input_seqs = [
        "<AA2fold> " + seq_1d,
        "<fold2AA> " + seq_3di,
    ]

    ids = tokenizer.batch_encode_plus(
        input_seqs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(
            ids.input_ids, attention_mask=ids.attention_mask
        )

    emb_aa = outputs.last_hidden_state[0, 1 : d1 + 1]
    emb_3di = outputs.last_hidden_state[1, 1 : d2 + 1]

    L = min(d1, d2)
    emb = torch.cat(
        [emb_aa[:L], emb_3di[:L]], dim=-1
    ).float()

    # 2048 → 512 projection
    proj = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
    ).to(device)

    with torch.no_grad():
        emb = proj(emb)

    return emb.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract TF embeddings using ProstT5 (AA + 3Di)"
    )
    parser.add_argument(
        "--aa_dir", required=True, help="Directory with AA FASTA files"
    )
    parser.add_argument(
        "--di_fasta", required=True, help="Foldseek 3Di FASTA file eg. ../3di_out/pdb_3Di_ss.fasta"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory"
    )
    parser.add_argument(
        "--device", default="cuda", help="cuda or cpu"
    )

    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all 3Di sequences
    di_dict = {
        rec.id.split()[0]: str(rec.seq)
        for rec in SeqIO.parse(args.di_fasta, "fasta")
    }

    for fname in os.listdir(args.aa_dir):
        if not fname.endswith(".fasta"):
            continue

        tf_id = fname.replace(".fasta", "")
        aa_path = os.path.join(args.aa_dir, fname)

        if tf_id not in di_dict:
            print(f"⚠️  No 3Di for {tf_id}, skipping")
            continue

        aa_seq = str(next(SeqIO.parse(aa_path, "fasta")).seq)
        di_seq = di_dict[tf_id]

        emb = embedding_features(aa_seq, di_seq, device)

        out_path = os.path.join(
            args.out_dir, f"{tf_id}_embedding.pt"
        )
        torch.save(emb, out_path)

        print(f" Saved {tf_id}: {tuple(emb.shape)} → {out_path}")


if __name__ == "__main__":
    main()



"""

nohup python extract_tf_embeddings.py \
  --aa_dir ../TFBindFormer/data/tf_data/tf_aa_sequence \
  --di_fasta ../TFBindFormer/data/tf_data/3di_out/pdb_3Di_ss.fasta \
  --out_dir ../TFBindFormer/data/tf_embeddings/
 > extract_tf_embeddings.log 2>&1 &


"""
