# models/tbinet_dna_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TBiNetDNAEncoder200(nn.Module):
    """
    TBiNet-like DNA encoder:
      [B,1000,4] --Conv(320,k=26)--> [B,320,975]
                   --MaxPool(13)--> [B,320,75]
              --1x1 attention-->   [B,320,75] (reweighted)
              --BiLSTM(320)-->     [B,75,640]
              --Linear->d_model--> [B,75,d_model]
              --Interpolate->200-->[B,200,d_model]
    """
    def __init__(
        self,
        d_model: int = 128,
        conv_filters: int = 320,
        conv_kernel: int = 26,
        pool_size: int = 13,
        lstm_hidden: int = 320,
        dropout: float = 0.2,
        add_posnorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.add_posnorm = add_posnorm

        # 1) Conv block (channels-first for Conv1d)
        self.conv = nn.Conv1d(in_channels=4, out_channels=conv_filters, kernel_size=conv_kernel, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.drop_conv = nn.Dropout(dropout)

        # 2) Simple content attention over positions (1x1 conv → softmax over length)
        self.attn_logits = nn.Conv1d(conv_filters, 1, kernel_size=1, bias=True)
        # (softmax computed in forward over length dimension)

        # 3) BiLSTM over pooled sequence (length ≈ 75)
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            dropout=0.0,           # single layer -> internal dropout not applied
            batch_first=True,
            bidirectional=True
        )
        self.drop_lstm = nn.Dropout(dropout)

        # 4) Project to d_model
        self.proj = nn.Linear(2 * lstm_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

        # (optional) tiny positional encoding for stability
        self.pos_linear = nn.Linear(1, d_model) if add_posnorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1000, 4] one-hot DNA
        returns: [B, 200, d_model]
        """
        B = x.size(0)

        # ---- Conv → Pool ----
        # to [B,4,1000] for Conv1d
        x = x.transpose(1, 2)                         # [B,4,1000]
        x = self.drop_conv(self.act(self.conv(x)))    # [B,320,975]
        x = self.pool(x)                              # [B,320,75]   (since 975/13 = 75)

        # ---- Attention over positions ----
        # logits: [B,1,75] → softmax over L
        attn_logits = self.attn_logits(x)             # [B,1,75]
        attn = torch.softmax(attn_logits, dim=-1)     # [B,1,75]
        x = x * attn                                  # [B,320,75] reweighted

        # ---- BiLSTM ----
        x = x.transpose(1, 2)                         # [B,75,320]
        x, _ = self.lstm(x)                           # [B,75,640]
        x = self.drop_lstm(x)

        # ---- Project to d_model ----
        x = self.proj(x)                              # [B,75,d_model]

        # ---- Upsample length: 75 → 200 ----
        # interpolate over sequence length (treat features as channels)
        x = x.transpose(1, 2)                         # [B,d_model,75]
        #x = F.interpolate(x, size=200, mode="linear", align_corners=False)  # [B,d_model,200]
        x = F.interpolate(x, size=200, mode="nearest")
        x = x.transpose(1, 2)                         # [B,200,d_model]

        # ---- (optional) add simple positional bias then LayerNorm ----
        if self.pos_linear is not None:
            # positions 0..199 normalized to [0,1]
            pos = torch.linspace(0, 1, 200, device=x.device).view(1, 200, 1).repeat(B, 1, 1)  # [B,200,1]
            x = x + self.pos_linear(pos)

        x = self.norm(x)                              # [B,200,d_model]
        return x
