import torch
import torch.nn as nn

from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import IntAttention, PatchSampling, CointAttention


class Model(nn.Module):
    """
    TimeBridge with optional embedding export.

    Default behavior (return_hidden=False):
      forward(...) -> pred  [B, pred_len, C]  (same as your original)

    If return_hidden=True:
      forward(...) -> (pred, hidden_dict)
        hidden_dict:
          tok_short: [B, C, N, D]   # after IA stack
          tok_long : [B, C, M, D]   # after PD(+CA) stack (or same as tok_short if no PD/CA)
          h_short  : [B, C, D]      # mean pool over tokens
          h_long   : [B, C, D]
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # kept for compatibility (you didn't use it in original code)
        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # base patch count from period
        self.num_p = self.seq_len // self.period

        # if configs.num_p is None, keep original behavior: set to self.num_p
        # NOTE: If you want real downsampling, you should pass --num_p smaller than self.num_p.
        if configs.num_p is None:
            configs.num_p = self.num_p

        # Patch embedding
        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        # Build layers explicitly so we can capture intermediate embeddings
        self.ia_layers = nn.ModuleList(self._build_integrated_attention(configs))
        self.pd_layers = nn.ModuleList(self._build_patch_sampling(configs))
        self.ca_layers = nn.ModuleList(self._build_cointegrated_attention(configs))

        # Decoder: identical to your original logic
        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),                         # [B,C,out_p,D] -> [B,C,out_p*D]
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)  # -> [B,C,pred_len]
        )

    # -------------------------
    # Layer builders (same as your original)
    # -------------------------
    def _build_integrated_attention(self, configs):
        layers = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout,
            stable_len=configs.stable_len,
            activation=configs.activation,
            stable=True,
            enc_in=self.c_in
        ) for _ in range(configs.ia_layers)]
        return layers

    def _build_patch_sampling(self, configs):
        layers = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            stable=False,
            stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p,
            out_p=configs.num_p,
            dropout=configs.dropout,
            activation=configs.activation
        ) for i in range(configs.pd_layers)]
        return layers

    def _build_cointegrated_attention(self, configs):
        layers = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            stable=False,
            enc_in=self.c_in,
            stable_len=configs.stable_len,
        ) for _ in range(configs.ca_layers)]
        return layers

    # -------------------------
    # Core forward (forecast)
    # -------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, return_hidden: bool = False):
        """
        x_enc:      [B, seq_len, C]
        x_mark_enc: [B, seq_len, time_feat] or None

        Returns:
          - pred: [B, pred_len, C]
          - optionally hidden dict
        """
        if x_mark_enc is None:
            # keep original behavior: default 4 time features
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        # same normalization as your original (per-sample, over time dim)
        mean = x_enc.mean(1, keepdim=True).detach()
        std = x_enc.std(1, keepdim=True).detach()
        x_norm = (x_enc - mean) / (std + 1e-5)

        # embedding -> tokens
        # PatchEmbed returns [B, C_total, N, D]; we only keep first C channels
        x_tok = self.embedding(x_norm, x_mark_enc)           # [B, C_total, N, D]
        x_tok = x_tok[:, :self.c_in, ...].contiguous()       # [B, C, N, D]

        # ---- IA stack ----
        x = x_tok
        for layer in self.ia_layers:
            x, _ = layer(x)
        tok_short = x  # [B,C,N,D]  (embedding after IA)

        # ---- PD stack ----
        for layer in self.pd_layers:
            x, _ = layer(x)
        # After PD, token count becomes configs.num_p (out_p) if pd_layers>0
        tok_after_pd = x

        # ---- CA stack (if enabled later) ----
        for layer in self.ca_layers:
            x, _ = layer(x)
        tok_long = x  # [B,C,M,D] (after PD + CA) ; if no PD/CA, equals tok_short

        # Decoder uses FINAL tokens (same as original: enc_out -> decoder)
        enc_out = tok_long
        dec_out = self.decoder(enc_out).transpose(-1, -2)    # [B, pred_len, C]
        pred = dec_out * std + mean                          # de-normalize

        if not return_hidden:
            return pred

        # pooled embeddings per stock
        h_short = tok_short.mean(dim=2)  # [B,C,D]
        h_long = tok_long.mean(dim=2)    # [B,C,D]

        hidden = {
            "tok_short": tok_short,
            "tok_long": tok_long,
            "h_short": h_short,
            "h_long": h_long,
        }
        return pred, hidden

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_hidden: bool = False):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_hidden=return_hidden)
        if return_hidden:
            pred, hidden = out
            return pred[:, -self.pred_len:, :], hidden
        else:
            pred = out
            return pred[:, -self.pred_len:, :]  # [B, pred_len, C]
