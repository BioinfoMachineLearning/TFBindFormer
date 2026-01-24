import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# ===========================================================
class ProteinReduceVariable(nn.Module):
    """
    If Lp <= target_len:
        project to d_model and keep length Lp: (B, Lp, d_model)
    If Lp > target_len:
        attention-reduce to target_len:       (B, target_len, d_model)

    Inputs:
        protein_emb:  (B, Lp, protein_in_dim)
        protein_mask: (B, Lp) bool, True = PAD (optional)

    Outputs:
        protein_rep:  (B, L_prot_out, d_model), where L_prot_out <= target_len
        prot_mask:    (B, L_prot_out) or None
    """

    def __init__(
        self,
        protein_in_dim: int = 512,
        d_model: int = 128,
        target_len: int = 200,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_len = target_len
        self.d_model = d_model

        # 512 -> 256 -> 128 projection with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(protein_in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # learned queries for reduction when Lp > target_len
        self.query = nn.Parameter(
            torch.randn(1, target_len, d_model) * (d_model ** -0.5)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,         # (B,Lp,512)
        protein_mask: torch.Tensor | None = None,  # (B,Lp) or None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # Project to d_model
        protein_emb = self.input_proj(protein_emb)   # (B,Lp,128)
        B, Lp, D = protein_emb.shape


        # --- Case 2: long protein, reduce to target_len via attention ---
        q = self.query.expand(B, -1, -1)  # (B,target_len,128)

        out, _ = self.attn(
            query=q,
            key=protein_emb,
            value=protein_emb,
            key_padding_mask=protein_mask,  # (B,Lp) or None
            need_weights=False,
        )
        out = self.norm(out + self.ff(out))  # (B,target_len,128)

        # after reduction, all positions are "real" latent tokens → no pad
        return out, None   # (B,target_len,128), mask=None


# ===========================================================
# FFN block used inside cross-attention encoder
# ===========================================================
class FFNBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


# ===========================================================
# Hybrid DNA ↔ Protein Cross-Attention Encoder
# ===========================================================
class HybridCrossAttentionEncoder(nn.Module):
    """
    Cross-attention encoder between DNA tokens and protein tokens.

    Inputs:
        protein:      (B, L_prot, d_model)  (L_prot can vary per batch)
        dna:          (B, L_dna, d_model)
        protein_mask: (B, L_prot) bool or None
        dna_mask:     (B, L_dna) bool or None

    Outputs:
        dna_ctx:  (B, L_dna, d_model)
        prot_ctx: (B, L_prot, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        num_bidir_layers: int = 2,
        dropout: float = 0.2,
        
    ):
        super().__init__()

        self.num_bidir_layers = num_bidir_layers

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "dna_to_prot_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                ),
                "prot_to_dna_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm_dna": nn.LayerNorm(d_model),
                "norm_prot": nn.LayerNorm(d_model),
                "ffn_dna": FFNBlock(d_model, dropout),
                "ffn_prot": FFNBlock(d_model, dropout),
            })
            for _ in range(num_layers)
        ])

        # optional logging
        self.last_dna_to_prot_attn = None  # (B, L_dna, L_prot)
        self.last_prot_to_dna_attn = None  # (B, L_prot, L_dna)

    def forward(
        self,
        protein: torch.Tensor,         # (B,L_prot,d_model)
        dna: torch.Tensor,             # (B,L_dna,d_model)
        protein_mask: torch.Tensor | None = None,  # (B,L_prot) or None
        dna_mask: torch.Tensor | None = None,      # (B,L_dna) or None
        return_both: bool = True,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:

        prot_ctx = protein
        dna_ctx = dna
        dna_to_prot_attn_maps = []
        prot_to_dna_attn_maps = []


        for layer_idx, layer in enumerate(self.layers):
            #need_w = self.enable_attention_logging

            # --- DNA → Protein attention ---
            dna_out, w_d2p = layer["dna_to_prot_attn"](
                query=dna_ctx,
                key=prot_ctx,
                value=prot_ctx,
                key_padding_mask=protein_mask,         # mask over protein length
                need_weights=return_attention,
                average_attn_weights=False,
            )
            if return_attention:
                dna_to_prot_attn_maps.append(w_d2p.detach())
                 

            dna_ctx = layer["norm_dna"](dna_ctx + dna_out)
            dna_ctx = layer["ffn_dna"](dna_ctx)

            # --- Protein → DNA attention (for early layers) ---
            if layer_idx < self.num_bidir_layers:
                prot_out, w_p2d = layer["prot_to_dna_attn"](
                    query=prot_ctx,
                    key=dna_ctx,
                    value=dna_ctx,
                    key_padding_mask=dna_mask,         # mask over DNA length (if any)
                    need_weights=return_attention,
                    average_attn_weights=False,
                )
                if return_attention:
                    prot_to_dna_attn_maps.append(w_p2d.detach())


                prot_ctx = layer["norm_prot"](prot_ctx + prot_out)
                prot_ctx = layer["ffn_prot"](prot_ctx)

        if return_attention:
            return dna_ctx, prot_ctx, {
                "dna_to_prot": dna_to_prot_attn_maps,
                "prot_to_dna": prot_to_dna_attn_maps
            }

        return (dna_ctx, prot_ctx) if return_both else dna_ctx

