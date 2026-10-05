# ============================================================
# models/binding_predictor.py
# ============================================================

import torch
import torch.nn as nn

from src.architectures.tbinet_dna_encoder import TBiNetDNAEncoder200
from src.architectures.cross_attention_encoder import HybridCrossAttentionEncoder
from src.architectures.cross_attention_encoder import ProteinReduceVariable

def reverse_complement_onehot(dna):
    """
    dna: torch.Tensor, shape (B, L, 4)
    column order: A, G, C, T
    """
    return torch.flip(dna, dims=[1])[:, :, [3, 2, 1, 0]]


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
        protein_in_dim: int = 1024,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.3,
        num_layers: int = 3,
        num_bidir_layers: int = 2,
        use_cell_type: bool = False,
        num_cell_types: int = 0,
        cell_type_dim: int = 16,
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

        # 2) protein reduction
 
        self.protein_reduce = ProteinReduceVariable(
            protein_in_dim=protein_in_dim,
            d_model=d_model,
            target_len=200,
            nhead=nhead,
            dropout=dropout,
            reduce_long=True,
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
        self.use_cell_type = use_cell_type
        self.cell_type_dim = cell_type_dim

        if self.use_cell_type:
            assert num_cell_types > 0, "num_cell_types must be > 0 when use_cell_type=True"
            self.cell_type_embedding = nn.Embedding(num_cell_types, cell_type_dim)
            classifier_in_dim = d_model + cell_type_dim
        else:
            self.cell_type_embedding = None
            classifier_in_dim = d_model

        # 5) classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in_dim),
            nn.Linear(classifier_in_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        dna_onehot: torch.Tensor,
        protein_emb: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
        dna_mask: torch.Tensor | None = None,
        cell_type_id: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:

        dna_rc = reverse_complement_onehot(dna_onehot)

        # ----- 1. Encode DNA -----
        dna_rep_fwd = self.dna_encoder(dna_onehot)
        dna_rep_rc = self.dna_encoder(dna_rc)

        # align RC embedding back to original forward coordinate
        dna_rep_rc = torch.flip(dna_rep_rc, dims=[1])

        #dna_rep = 0.5 * (dna_rep_fwd + dna_rep_rc)  # (B,L_dna=200,d_model)
        dna_rep = torch.maximum(dna_rep_fwd, dna_rep_rc)
        # ----- 2. Reduce / project protein -----
        protein_rep, prot_mask = self.protein_reduce(
            protein_emb, protein_mask
        )
        # protein_rep: (B,L_prot_out,d_model), L_prot_out <= 200
        # prot_mask:   (B,L_prot_out) or None
       
        if return_attention:
            dna_out, prot_out, attn = self.cross_encoder(
                protein=protein_rep,
                dna=dna_rep,
                protein_mask=prot_mask,
                dna_mask=dna_mask,
                return_both=True,
                return_attention=True,
            )
        else:
            dna_out, prot_out = self.cross_encoder(
                protein=protein_rep,
                dna=dna_rep,
                protein_mask=prot_mask,
                dna_mask=dna_mask,
                return_both=True,
                return_attention=False,
            )




        # ----- 4. Position-weighted pooling over DNA -----
        pooled = self.pool(dna_out, mask=dna_mask)   # (B,d_model)
        if self.use_cell_type:
            if cell_type_id is None:
                raise ValueError("cell_type_id must be provided when use_cell_type=True")

            cell_emb = self.cell_type_embedding(cell_type_id)  # (B, cell_type_dim)
            pooled = torch.cat([pooled, cell_emb], dim=-1)

        # ----- 5. Classifier -----
        logits = self.classifier(pooled).squeeze(-1)

        if return_attention:
            return logits, attn
        return logits