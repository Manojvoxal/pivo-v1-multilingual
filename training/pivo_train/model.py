"""PyTorch re-implementation of the PiVo-1.4M acoustic model.

Rebuilt from the released ONNX graphs (encoder.onnx / decoder.onnx) and the
`acoustic.pt` checkpoint: parameter names and shapes match the checkpoint's
`state_dict` exactly (584,965 parameters + feature_mean/std buffers), so
`load_state_dict(strict=True)` works. `tests/test_parity.py` checks the outputs
against the official ONNX files.

Architecture (width 96, 4 query heads / 2 kv heads of 24 dims, RMSNorm eps 1e-6):
  encoder : token emb + voice emb -> 6 blocks alternating
              attention block  x += Attn(RMSNorm(x)); x += SwiGLU(RMSNorm(x))   (RoPE, QK-norm, GQA)
              conv block       x += SiLU(RMSNorm(Conv1d_k5_dil{1,2,4}(RMSNorm(x))))
            -> RMSNorm -> encoded
  durations: in_proj+SiLU -> 2 x [depthwise conv3 -> pointwise conv -> SiLU -> RMSNorm] -> linear
             predicts log(1 + frames) per token
  decoder : encoded repeated by durations + Linear(4 phase features) ->
            2 causal attention blocks (window 64 frames; RoPE only in block 0)
            -> RMSNorm -> Linear(20) = normalized features; * std + mean = Opus/FARGAN features
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "width": 96, "ff_width": 192, "heads": 4, "kv_heads": 2, "text_layers": 6,
    "decoder_layers": 2, "kernel": 5, "duration_width": 64, "feature_dim": 20,
    "attention_window": 64, "dropout": 0.1, "max_token_frames": 200,
    "max_utterance_frames": 3000, "vocab_size": 128, "num_voices": 2,
}
EPS = 1e-6
ROPE_BASE = 10000.0
MASK_VALUE = -10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return xf.to(x.dtype) * self.weight


def rope(x, positions):
    """x: [B, H, T, D]; positions: [T] or [B, T] (float). Half-split rotation, D/2 freqs."""
    d = x.shape[-1]
    inv_freq = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=torch.float32) * (-math.log(ROPE_BASE) / d))
    ang = positions.float()[..., None] * inv_freq                       # [..., T, D/2]
    while ang.dim() < 4:
        ang = ang.unsqueeze(-3) if ang.dim() == 3 else ang.unsqueeze(0)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        self.q = nn.Linear(width, heads * self.head_dim, bias=False)
        self.k = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def project(self, x, positions=None):
        b, t, _ = x.shape
        q = self.q_norm(self.q(x).view(b, t, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(b, t, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if positions is not None:
            q, k = rope(q, positions), rope(k, positions)
        return q, k, v

    def attend(self, q, k, v, allowed):
        """allowed: bool, broadcastable to [B, H, Tq, Tk]."""
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, MASK_VALUE)
        out = scores.softmax(-1) @ v
        b, h, t, d = out.shape
        return self.out(out.transpose(1, 2).reshape(b, t, h * d))


class AttentionBlock(nn.Module):
    def __init__(self, c, dropout):
        super().__init__()
        self.norm1 = RMSNorm(c["width"])
        self.attention = Attention(c["width"], c["heads"], c["kv_heads"])
        self.norm2 = RMSNorm(c["width"])
        self.gate = nn.Linear(c["width"], c["ff_width"], bias=False)
        self.up = nn.Linear(c["width"], c["ff_width"], bias=False)
        self.down = nn.Linear(c["ff_width"], c["width"], bias=False)
        self.dropout = nn.Dropout(dropout)

    def ff(self, x):
        h = self.norm2(x)
        return self.dropout(self.down(F.silu(self.gate(h)) * self.up(h)))

    def forward(self, x, allowed, positions=None):
        q, k, v = self.attention.project(self.norm1(x), positions)
        x = x + self.dropout(self.attention.attend(q, k, v, allowed))
        return x + self.ff(x)


class ConvBlock(nn.Module):
    def __init__(self, c, dilation, dropout):
        super().__init__()
        self.norm1 = RMSNorm(c["width"])
        self.conv = nn.Conv1d(c["width"], c["width"], c["kernel"], dilation=dilation,
                              padding=dilation * (c["kernel"] - 1) // 2, bias=False)
        self.norm2 = RMSNorm(c["width"])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):  # mask: [B, T, 1] float
        h = self.norm1(x) * mask
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return (x + self.dropout(F.silu(self.norm2(h)))) * mask


class DurationPredictor(nn.Module):
    def __init__(self, c, dropout):
        super().__init__()
        w = c["duration_width"]
        self.in_proj = nn.Linear(c["width"], w)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(w, w, 3, padding=1, groups=w), nn.Conv1d(w, w, 1), nn.SiLU())
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([RMSNorm(w) for _ in range(2)])
        self.out = nn.Linear(w, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = F.silu(self.in_proj(x)) * mask
        for layer, norm in zip(self.layers, self.norms):
            h = norm(layer(h.transpose(1, 2)).transpose(1, 2)) * mask
            h = self.dropout(h)
        return self.out(h).squeeze(-1) * mask.squeeze(-1)


def phase_features(durations, max_token_frames=200):
    """Torch port of pivo.phase.phoneme_phase_features for a 1-D LongTensor of durations."""
    durations = durations.long()
    counts = torch.repeat_interleave(durations, durations).float()
    starts = torch.repeat_interleave(torch.cumsum(durations, 0) - durations, durations).float()
    frac = (torch.arange(len(counts), device=durations.device, dtype=torch.float32) - starts + 0.5) / counts
    return torch.stack((2 * frac - 1, torch.sin(math.pi * frac), torch.cos(math.pi * frac),
                        torch.log1p(counts) / math.log1p(max_token_frames)), dim=-1)


class PiVoAcoustic(nn.Module):
    def __init__(self, config=None, num_voices=None):
        super().__init__()
        c = dict(CONFIG, **(config or {}))
        if num_voices is not None:
            c["num_voices"] = num_voices
        self.config = c
        d = c["dropout"]
        self.register_buffer("feature_mean", torch.zeros(c["feature_dim"]))
        self.register_buffer("feature_std", torch.ones(c["feature_dim"]))
        self.tokens = nn.Embedding(c["vocab_size"], c["width"])
        self.voices = nn.Embedding(c["num_voices"], c["width"])
        blocks = []
        dilation = 1
        for i in range(c["text_layers"]):
            if i % 2 == 0:
                blocks.append(AttentionBlock(c, d))
            else:
                blocks.append(ConvBlock(c, dilation, d))
                dilation *= 2
        self.text_blocks = nn.ModuleList(blocks)
        self.text_norm = RMSNorm(c["width"])
        self.durations = DurationPredictor(c, d)
        self.decoder = nn.ModuleList([AttentionBlock(c, d) for _ in range(c["decoder_layers"])])
        self.feature_norm = RMSNorm(c["width"])
        self.feature_head = nn.Linear(c["width"], c["feature_dim"])
        self.phase_projection = nn.Linear(4, c["width"], bias=False)

    # ------------------------------------------------------------------ encoder
    def encode(self, tokens, voice, lengths):
        """tokens [B,T] long, voice [B] long, lengths [B] long -> (encoded [B,T,96], log_durations [B,T])."""
        b, t = tokens.shape
        valid = torch.arange(t, device=tokens.device)[None] < lengths[:, None]       # [B,T]
        mask = valid[..., None].float()
        allowed = valid[:, None, None, :]                                               # key padding
        positions = torch.arange(t, device=tokens.device)
        x = (self.tokens(tokens) + self.voices(voice)[:, None]) * mask
        for block in self.text_blocks:
            if isinstance(block, AttentionBlock):
                x = block(x, allowed, positions) * mask
            else:
                x = block(x, mask)
        encoded = self.text_norm(x) * mask
        return encoded, self.durations(encoded, mask)

    # ------------------------------------------------------------------ decoder (training, full sequence)
    def expand(self, encoded, durations, frame_lengths):
        """Repeat token encodings by durations and append phase features -> [B, F, 100].
        Fully vectorised over the batch (no Python loop)."""
        b, t, w = encoded.shape
        f_max = int(frame_lengths.max())
        dur = durations.long().clamp(min=0)
        flat = dur.reshape(-1)
        total = int(flat.sum())
        out = encoded.new_zeros(b, f_max, w + 4)
        if total == 0:
            return out
        tok_idx = torch.repeat_interleave(torch.arange(b * t, device=encoded.device), flat)   # [total]
        batch_idx = tok_idx // t
        per_item = dur.sum(1)
        item_start = torch.cumsum(per_item, 0) - per_item
        frame_idx = torch.arange(total, device=encoded.device) - item_start[batch_idx]
        counts = flat[tok_idx].float()
        tok_start = (torch.cumsum(flat, 0) - flat)[tok_idx]
        frac = (torch.arange(total, device=encoded.device) - tok_start).float() + 0.5
        frac = frac / counts
        phase = torch.stack((2 * frac - 1, torch.sin(math.pi * frac), torch.cos(math.pi * frac),
                             torch.log1p(counts) / math.log1p(self.config["max_token_frames"])), dim=-1)
        out[batch_idx, frame_idx] = torch.cat((encoded.reshape(b * t, w)[tok_idx], phase.to(encoded.dtype)), -1)
        return out

    def decode(self, expanded):
        """expanded [B, F, 100] -> normalized features [B, F, 20] (causal, window 64)."""
        f = expanded.shape[1]
        x = expanded[..., :-4] + self.phase_projection(expanded[..., -4:])
        pos = torch.arange(f, device=expanded.device)
        delta = pos[:, None] - pos[None, :]                                            # query - key
        allowed = ((delta >= 0) & (delta < self.config["attention_window"]))[None, None]
        for i, block in enumerate(self.decoder):
            x = block(x, allowed, pos if i == 0 else None)
        return self.feature_head(self.feature_norm(x))

    def forward(self, tokens, voice, token_lengths, durations, frame_lengths):
        encoded, log_durations = self.encode(tokens, voice, token_lengths)
        expanded = self.expand(encoded, durations, frame_lengths)
        return self.decode(expanded), log_durations

    def denormalize(self, normalized):
        return normalized * self.feature_std + self.feature_mean

    # ------------------------------------------------------------------ inference helpers
    @torch.no_grad()
    def predict_durations(self, log_durations, tokens, speed=1.0):
        c = self.config
        d = torch.round(torch.expm1(log_durations.clamp(0, math.log1p(c["max_token_frames"]))) / speed)
        d = d.clamp(0, c["max_token_frames"]).long()
        d = torch.maximum(d, (tokens >= 10).long())   # every phone gets >= 1 frame
        d[tokens <= 2] = 0                            # <pad>, <bos>, <eos>
        return d

    @torch.no_grad()
    def synthesize_features(self, tokens, voice, speed=1.0):
        """tokens: 1-D LongTensor for one utterance -> (features [F,20] denormalized, durations)."""
        tokens = tokens[None]
        enc, logd = self.encode(tokens, torch.tensor([voice], device=tokens.device),
                                torch.tensor([tokens.shape[1]], device=tokens.device))
        dur = self.predict_durations(logd, tokens, speed)[0]
        expanded = self.expand(enc, dur[None], dur.sum()[None])
        return self.denormalize(self.decode(expanded))[0], dur

    # ------------------------------------------------------------------ streaming step (matches decoder.onnx)
    def decode_step(self, expanded, position, caches):
        """expanded [1, T, 100]; position: int tensor (abs index of first frame);
        caches: [key0, value0, key1, value1], each [1, kv_heads, L, 24] (L <= window-1).
        Returns (features [1,T,20] denormalized, new caches). Mirrors decoder.onnx exactly."""
        t = expanded.shape[1]
        w = self.config["attention_window"]
        x = expanded[..., :-4] + self.phase_projection(expanded[..., -4:])
        q_pos = position + torch.arange(t, device=x.device)
        new = []
        for i, block in enumerate(self.decoder):
            kc, vc = caches[2 * i], caches[2 * i + 1]
            q, k, v = block.attention.project(block.norm1(x), q_pos if i == 0 else None)
            k = torch.cat((kc, k), dim=2)
            v = torch.cat((vc, v), dim=2)
            n_keys = k.shape[2]
            k_pos = position + t - n_keys + torch.arange(n_keys, device=x.device)
            delta = q_pos[:, None] - k_pos[None, :]
            allowed = ((delta >= 0) & (delta < w) & (k_pos[None, :] >= 0))[None, None]
            x = x + block.attention.attend(q, k, v, allowed)
            x = x + block.ff(x)
            new += [k[:, :, -(w - 1):], v[:, :, -(w - 1):]]
        return self.denormalize(self.feature_head(self.feature_norm(x))), new


# ---------------------------------------------------------------------------- warm start
# Kannada phones (ids 79-126) start from the closest English phone's embedding, so the
# encoder already "roughly knows" how they sound. Order must match pivo_kn/symbols.py.
KN_TO_ARPABET = {
    "a": "AH0", "aa": "AA1", "i": "IH0", "ii": "IY1", "u": "UH0", "uu": "UW1", "e": "EH1",
    "ee": "EY1", "o": "OW0", "oo": "OW1", "ai": "AY1", "au": "AW1",
    "k": "K", "kh": "K", "g": "G", "gh": "G", "ng": "NG", "c": "CH", "ch": "CH", "j": "JH",
    "jh": "JH", "ny": "N", "tt": "T", "tth": "T", "dd": "D", "ddh": "D", "nn": "N",
    "t": "T", "th": "T", "d": "D", "dh": "D", "n": "N", "p": "P", "ph": "P", "b": "B",
    "bh": "B", "m": "M", "y": "Y", "r": "R", "l": "L", "v": "V", "ll": "L", "sh": "SH",
    "ssh": "SH", "s": "S", "h": "HH", "f": "F", "z": "Z",
}


def load_pivo_checkpoint(path, num_voices=4, voice_init=None, symbols=None, seed=0):
    """Build a model from PiVo's acoustic.pt.

    voice_init: list mapping each new voice id -> source PiVo voice (0 = Pi/F, 1 = Vo/M).
    symbols: full symbol list (manifests/symbols.json) to initialise Kannada token rows.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = PiVoAcoustic(config={k: v for k, v in ckpt.get("config", {}).items() if k in CONFIG},
                         num_voices=2)
    model.load_state_dict(state, strict=True)
    if num_voices != 2:
        old = model.voices.weight.data.clone()
        model.voices = nn.Embedding(num_voices, old.shape[1])
        voice_init = voice_init or [i % 2 for i in range(num_voices)]
        model.voices.weight.data = old[torch.tensor(voice_init)].clone()
        model.config["num_voices"] = num_voices
    if symbols is not None:
        sym_id = {s: i for i, s in enumerate(symbols)}
        g = torch.Generator().manual_seed(seed)
        emb = model.tokens.weight.data
        scale = emb[10:79].std().item()
        for s, i in sym_id.items():
            if s.startswith("kn_") and s[3:] in KN_TO_ARPABET:
                src = sym_id[KN_TO_ARPABET[s[3:]]]
                emb[i] = emb[src] + 0.1 * scale * torch.randn(emb.shape[1], generator=g)
    return model, ckpt
