import re
import torch
from transformers import T5Tokenizer, T5EncoderModel

class ProstT5Embedder(torch.nn.Module):
    """
    Outputs FIXED 512-dim per-residue embeddings.
    """

    def __init__(self, device=None):
        super().__init__()

        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/ProstT5", do_lower_case=False
        )

        self.model = T5EncoderModel.from_pretrained(
            "Rostlab/ProstT5"
        ).to(self.device)

        if self.device.type == "cuda":
            self.model.half()
        else:
            self.model.float()

        # ✅ projections moved to device
        self.proj_single = torch.nn.Sequential(
            torch.nn.Linear(1024, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 512),
        ).to(self.device)

        self.proj_joint = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
        ).to(self.device)

        # freeze everything
        for p in self.parameters():
            p.requires_grad = False

        if self.device.type == "cuda":
            self.model.half()
            self.proj_single.half()
            self.proj_joint.half()
        else:
            self.model.float()
            self.proj_single.float()
            self.proj_joint.float()

        self.eval()

    def _encode(self, seq, token):
        seq = " ".join(seq)
        ids = self.tokenizer(
            f"{token} {seq}",
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            ).last_hidden_state

        L = len(seq.split())
        return out[0, 1:L+1]

    def forward(self, seq_1d=None, seq_3di=None, mode="joint"):
        with torch.no_grad():

            if mode == "aa":
                seq_1d = re.sub(r"[UZOB]", "X", seq_1d)
                emb = self._encode(seq_1d, "<AA2fold>")
                out = self.proj_single(emb)
                return out.float().cpu()

            if mode == "3di":
                seq_3di = seq_3di.lower()
                emb = self._encode(seq_3di, "<fold2AA>")
                out = self.proj_single(emb)
                return out.float().cpu()

            if mode == "joint":
                seq_1d = re.sub(r"[UZOB]", "X", seq_1d)
                seq_3di = seq_3di.lower()

                emb_aa = self._encode(seq_1d, "<AA2fold>")
                emb_3di = self._encode(seq_3di, "<fold2AA>")

                L = min(len(emb_aa), len(emb_3di))
                emb = torch.cat([emb_aa[:L], emb_3di[:L]], dim=-1)

                out = self.proj_joint(emb)
                return out.float().cpu()

            raise ValueError(f"Unknown mode: {mode}")
