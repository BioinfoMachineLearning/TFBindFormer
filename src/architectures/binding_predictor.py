# ============================================================
# models/binding_predictor.py
# ============================================================

import torch
import torch.nn as nn

from src.architectures.tbinet_dna_encoder import TBiNetDNAEncoder200
from src.architectures.cross_attention_encoder import HybridCrossAttentionEncoder
from src.architectures.cross_attention_encoder import ProteinReduceVariable


# ===========================================================
# Position-Weighted Pooling over DNA
# ===========================================================
class PositionWeightedPool(nn.Module):
    """
    Learnable position-weighted pooling over DNA tokens.

    Input:
        x:    (B, L_dna, d_model)
        mask: (B, L_dna) bool or None   (True = pad)

    Output:
        pooled: (B, d_model)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.pos_score = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # x: (B,L,D)
        scores = self.pos_score(x).squeeze(-1)  # (B,L)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=1)     # (B,L)

        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # (B,D)
        return pooled


# ===========================================================
# Final DNA–Protein Binding Predictor
# ===========================================================
class DNABindingPredictor(nn.Module):
    """
    Full TF–DNA binding model:

      DNA encoder:
         (B,1000,4)   → (B, L_dna=200, d_model)
      Protein reduction:
         (B,Lp,512)   → (B, L_prot_out<=200, d_model)
      Cross attention:
         DNA(L_dna) ↔ Protein(L_prot_out)
      Position-weighted pooling over DNA:
         (B,L_dna,d_model) → (B,d_model)
      Classifier:
         (B,d_model) → (B,)

    Args:
        protein_in_dim: raw protein embedding dimension (e.g., 512)
        d_model:        transformer hidden size (e.g., 128)
    """

    def __init__(
        self,
        protein_in_dim: int = 512,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.3,
        num_layers: int = 3,
        num_bidir_layers: int = 2,
    ):
        super().__init__()

        # 1) DNA encoder → (B,L_dna=200,d_model)
        self.dna_encoder = TBiNetDNAEncoder200(
            d_model=d_model,
            conv_filters=320,
            conv_kernel=26,
            pool_size=13,
            lstm_hidden=320,
            dropout=0.2,
            add_posnorm=True,
        )

        # 2) protein reduction (pdb length → =200)
        self.protein_reduce = ProteinReduceVariable(
            protein_in_dim=protein_in_dim,
            d_model=d_model,
            target_len=200,
            nhead=nhead,
            dropout=dropout,
        )

        # 3) cross attention encoder
        self.cross_encoder = HybridCrossAttentionEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_bidir_layers=num_bidir_layers,
            dropout=dropout,

        )

        # 4) DNA pooling
        self.pool = PositionWeightedPool(d_model=d_model)

        # 5) classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        dna_onehot: torch.Tensor,          # (B,1000,4)
        protein_emb: torch.Tensor,         # (B,Lp,512)
        protein_mask: torch.Tensor | None = None,  # (B,Lp) bool or None
        dna_mask: torch.Tensor | None = None,      # (B,200) bool or None (optional)
        return_attention: bool = False,
    ) -> torch.Tensor:

        # ----- 1. Encode DNA -----
        dna_embed = self.dna_encoder(dna_onehot)   # (B,L_dna=200,d_model)

        # ----- 2. Reduce / project protein -----
        protein_rep, prot_mask = self.protein_reduce(
            protein_emb, protein_mask
        )
        # protein_rep: (B,L_prot_out,d_model), L_prot_out <= 200
        # prot_mask:   (B,L_prot_out) or None
       
        if return_attention:
            dna_out, prot_out, attn = self.cross_encoder(
                protein=protein_rep,
                dna=dna_embed,
                protein_mask=prot_mask,
                dna_mask=dna_mask,
                return_both=True,
                return_attention=True,
            )
        else:
            dna_out, prot_out = self.cross_encoder(
                protein=protein_rep,
                dna=dna_embed,
                protein_mask=prot_mask,
                dna_mask=dna_mask,
                return_both=True,
                return_attention=False,
            )




        # ----- 4. Position-weighted pooling over DNA -----
        pooled = self.pool(dna_out, mask=dna_mask)   # (B,d_model)

        # ----- 5. Classifier -----
        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if return_attention:
            return logits, attn
        return logits