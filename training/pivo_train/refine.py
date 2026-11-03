"""Stage 2: spectral refinement of a trained acoustic model (PiVo technical report, sec. 3.3).

Starts from a stage-1 checkpoint (e.g. runs/kn_en_v1/best.pt) and trains only the
decoder side against a small temporal critic, which counters the over-smoothing that the
smooth-L1 loss causes (a frequent source of the "metallic" sound).

Recipe as stated in the report:
  * 1,200 updates, batch size 4, 128-frame segments, learning rate 1e-4
  * two-scale temporal critic (~110k params) that sees the 18 cepstral channels + voice id
  * generator loss = supervised smooth-L1 + 0.01 * adversarial + 0.1 * feature matching
  * encoder and duration predictor frozen (61 tensors incl. the feature mean/std buffers),
    and we check afterwards that they did not change
  * the critic is only used for training; the exported model has the same size as before

The report doesn't specify these, so we use common choices: LSGAN losses (as in HiFi-GAN),
AdamW with betas (0.8, 0.99) for both nets, and a critic layout sized to about 110k params.

Writes to --out:
  samples/ceiling/voiceN_<spk>_kK_real.wav   real features -> vocoder (the vocoder's floor)
  samples/step_XXXXXX/voiceN_<spk>_kK_tf.wav  model with the real durations (same timing as real)
  samples/step_XXXXXX/voiceN_<spk>_kK_tts.wav normal TTS (predicted durations)
  refine_log.jsonl, step_XXXXXX.pt, refined.pt (= final step), last.pt (resume, has critic)

Step 0 is the input model before refinement, so A/B = step_000000 vs step_001200.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PiVoDataset, collate, read_manifest
from .model import PiVoAcoustic
from .train import batch_to

DECODER_PREFIXES = ("decoder.", "feature_norm.", "feature_head.", "phase_projection.")
N_CEPSTRAL = 18


# ----------------------------------------------------------------------------- critic
class ScaleCritic(nn.Module):
    """1-D conv critic over [B, 18, T] cepstra, conditioned on the voice by concatenating a
    voice embedding to the input channels. Returns (logits [B, 1, T'], feature maps)."""

    def __init__(self, num_voices, channels=58, voice_dim=19):
        super().__init__()
        c = channels
        self.voice = nn.Embedding(num_voices, voice_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(N_CEPSTRAL + voice_dim, c, 5, padding=2),
            nn.Conv1d(c, c, 5, stride=2, padding=2),
            nn.Conv1d(c, c, 5, padding=4, dilation=2),
            nn.Conv1d(c, c, 3, padding=1),
        ])
        self.out = nn.Conv1d(c, 1, 3, padding=1)

    def forward(self, x, voice):
        v = self.voice(voice)[:, :, None].expand(-1, -1, x.shape[-1])
        h = torch.cat((x, v), 1)
        feats = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)
            feats.append(h)
        return self.out(h), feats


class TemporalCritic(nn.Module):
    """Two scales: the 10 ms frame rate and a 2x average-pooled (20 ms) copy."""

    def __init__(self, num_voices, channels=58):
        super().__init__()
        self.scales = nn.ModuleList([ScaleCritic(num_voices, channels) for _ in range(2)])

    def forward(self, x, voice):  # x: [B, 18, T]
        outs = []
        for i, critic in enumerate(self.scales):
            xi = x if i == 0 else F.avg_pool1d(x, 2, 2)
            outs.append(critic(xi, voice))
        return outs


def d_loss_fn(real_outs, fake_outs):
    return sum(((r - 1) ** 2).mean() + (f ** 2).mean() for (r, _), (f, _) in zip(real_outs, fake_outs))


def g_losses(real_outs, fake_outs):
    adv = sum(((f - 1) ** 2).mean() for f, _ in fake_outs)
    fm, n = 0.0, 0
    for (_, rf), (_, ff) in zip(real_outs, fake_outs):
        for a, b in zip(rf, ff):
            fm = fm + F.l1_loss(b, a.detach())
            n += 1
    return adv, fm / max(n, 1)


# ----------------------------------------------------------------------------- helpers
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--init", required=True, help="stage-1 checkpoint, e.g. runs/kn_en_v1/best.pt")
    p.add_argument("--onnx-dir", default=None, help="PiVo onnx/ dir (vocoder for sample WAVs)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--segment", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--adv-weight", type=float, default=0.01)
    p.add_argument("--fm-weight", type=float, default=0.1)
    p.add_argument("--critic-channels", type=int, default=58)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-val-batches", type=int, default=50)
    p.add_argument("--samples-per-voice", type=int, default=3)
    p.add_argument("--pack-dir", default=None)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--resume", default="auto", help="'auto' = <out>/last.pt if present, 'none' = fresh")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def is_decoder(name):
    return name.startswith(DECODER_PREFIXES)


def set_mode(model):
    """Frozen encoder/duration predictor stay deterministic (eval); decoder trains (dropout on)."""
    model.eval()
    for m in (model.decoder, model.feature_norm, model.feature_head, model.phase_projection):
        m.train()


def masked_feature_loss(pred, target, frame_lengths):
    fmask = (torch.arange(pred.shape[1], device=pred.device)[None] < frame_lengths[:, None]).float()
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    return (loss * fmask).sum() / fmask.sum()


def random_segments(pred, target, frame_lengths, seg, gen):
    """Same random 128-frame window from prediction and target, cepstra only -> [B, 18, seg]."""
    p, t = [], []
    for i, n in enumerate(frame_lengths.tolist()):
        s = int(torch.randint(0, n - seg + 1, (1,), generator=gen)) if n > seg else 0
        p.append(pred[i, s:s + seg, :N_CEPSTRAL])
        t.append(target[i, s:s + seg, :N_CEPSTRAL])
    return torch.stack(p).transpose(1, 2), torch.stack(t).transpose(1, 2)


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    """Val smooth-L1 (real durations) and the report's smoothness number: how much the
    cepstra change frame to frame, prediction vs real (released PiVo: ~0.65; 1.0 = real)."""
    model.eval()
    tot, n, d_pred, d_real = 0.0, 0, 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = batch_to(batch, device)
        pred, _ = model(b["tokens"], b["voices"], b["token_lengths"], b["durations"], b["frame_lengths"],
                        lang=b.get("lang_ids") if model.langs is not None else None)
        tot += masked_feature_loss(pred, b["features"], b["frame_lengths"]).item()
        n += 1
        fl = b["frame_lengths"]
        pm = model.denormalize(pred)[..., :N_CEPSTRAL]
        rm = model.denormalize(b["features"])[..., :N_CEPSTRAL]
        valid = (torch.arange(pred.shape[1] - 1, device=pred.device)[None] < (fl - 1)[:, None]).float()[..., None]
        d_pred += ((pm[:, 1:] - pm[:, :-1]).abs() * valid).sum().item()
        d_real += ((rm[:, 1:] - rm[:, :-1]).abs() * valid).sum().item()
    set_mode(model)
    return {"val_feature": tot / max(n, 1), "smoothness": d_pred / max(d_real, 1e-9)}


def sample_rows(rows, per_voice):
    picked, count = [], {}
    for idx, r in enumerate(rows):
        v = r["voice_id"]
        if count.get(v, 0) < per_voice:
            picked.append(idx)
            count[v] = count.get(v, 0) + 1
    return picked


@torch.no_grad()
def write_samples(model, ds, picked, vocoder, out_dir, device, ceiling_dir=None):
    from .vocoder import write_wav
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    k_of = {}
    for idx in picked:
        r = ds.rows[idx]
        k = k_of[r["voice_id"]] = k_of.get(r["voice_id"], 0) + 1
        name = f"voice{r['voice_id']}_{r['speaker']}_k{k}"
        tokens = torch.tensor(r["token_ids"], device=device)
        feats, _ = model.synthesize_features(tokens, r["voice_id"],                       # normal TTS
                                             lang=r.get("lang_id") if model.langs is not None else None)
        write_wav(os.path.join(out_dir, name + "_tts.wav"), vocoder(feats.cpu().numpy()))
        item = collate([ds[idx]])
        b = batch_to(item, device)
        pred, _ = model(b["tokens"], b["voices"], b["token_lengths"], b["durations"], b["frame_lengths"],
                        lang=b.get("lang_ids") if model.langs is not None else None)
        tf = model.denormalize(pred)[0].cpu().numpy()                                     # real timing
        write_wav(os.path.join(out_dir, name + "_tf.wav"), vocoder(tf))
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(r["text"] + "\n")
        if ceiling_dir:
            os.makedirs(ceiling_dir, exist_ok=True)
            real = model.denormalize(b["features"])[0].cpu().numpy()                      # real features
            write_wav(os.path.join(ceiling_dir, name + "_real.wav"), vocoder(real))
            with open(os.path.join(ceiling_dir, name + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(r["text"] + "\n")
    set_mode(model)


def save_model(path, model, src, step, extra):
    tmp = path + ".tmp"
    keep = {k: src[k] for k in ("symbols", "voices", "voice_init", "pivo_parent", "languages") if k in src}
    torch.save({"kind": ("pivo-acoustic-v1-multi-refined" if model.langs is not None else "pivo-acoustic-v1-kn-refined"), "step": step, "config": model.config,
                "state_dict": model.state_dict(), **keep, **extra}, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main
def main(argv=None, on_checkpoint=None):
    a = parse_args(argv)
    hook = on_checkpoint or (lambda: None)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "refine_log.jsonl")

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    src = torch.load(a.init, map_location="cpu", weights_only=False)
    model = PiVoAcoustic(config=src["config"])
    model.load_state_dict(src["state_dict"], strict=True)
    model.to(a.device)
    frozen_ref = {k: v.clone() for k, v in src["state_dict"].items() if not is_decoder(k)}
    for name, prm in model.named_parameters():
        prm.requires_grad = is_decoder(name)
    g_params = [p for p in model.parameters() if p.requires_grad]
    print(f"init {a.init} (stage-1 step {src.get('step')}) | trainable decoder params "
          f"{sum(p.numel() for p in g_params):,} | frozen tensors {len(frozen_ref)}", flush=True)

    critic = TemporalCritic(model.config["num_voices"], a.critic_channels).to(a.device)
    print(f"critic params {sum(p.numel() for p in critic.parameters()):,} (report: 110,018)", flush=True)
    g_opt = torch.optim.AdamW(g_params, lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)
    d_opt = torch.optim.AdamW(critic.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)

    # data: same rows/filters as stage 1 so the one-file feature cache is reused
    train_rows = read_manifest(a.data_root, "train")
    val_rows = []
    for split in ("val", "test"):
        if os.path.exists(os.path.join(a.data_root, "manifests", f"{split}.jsonl")):
            val_rows = read_manifest(a.data_root, split)
            break
    if not val_rows:
        val_rows = train_rows[:max(1, len(train_rows) // 50)]
    mean, std = model.feature_mean.cpu().numpy(), model.feature_std.cpu().numpy()
    tag = os.path.basename(os.path.normpath(a.data_root))
    pack = lambda s: os.path.join(a.pack_dir, f"{tag}_{s}") if a.pack_dir else None
    t0 = time.time()
    train_ds = PiVoDataset(a.data_root, train_rows, mean, std, max_frames=model.config["max_utterance_frames"],
                           preload=a.preload, pack_path=pack("train"))
    val_ds = PiVoDataset(a.data_root, val_rows, mean, std, preload=a.preload, pack_path=pack("val"))
    hook()
    eligible = [i for i, r in enumerate(train_ds.rows) if r["n_frames"] >= a.segment]
    print(f"data: {len(eligible)} train clips >= {a.segment} frames, {len(val_ds)} val | loaded in "
          f"{time.time() - t0:.0f}s", flush=True)
    gen = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, eligible), batch_size=a.batch_size,
                              shuffle=True, drop_last=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate)
    picked = sample_rows(val_ds.rows, a.samples_per_voice)

    vocoder = None
    if a.onnx_dir:
        from .vocoder import Vocoder
        vocoder = Vocoder(a.onnx_dir)

    step = 0
    last = os.path.join(a.out, "last.pt")
    if a.resume == "auto" and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        critic.load_state_dict(ck["critic"])
        g_opt.load_state_dict(ck["g_opt"])
        d_opt.load_state_dict(ck["d_opt"])
        step = ck["step"]
        print(f"resumed from {last} at step {step}", flush=True)

    def checkpoint(final=False):
        save_model(os.path.join(a.out, f"step_{step:06d}.pt"), model, src, step, {"refine": vars(a)})
        if final:
            save_model(os.path.join(a.out, "refined.pt"), model, src, step, {"refine": vars(a)})
        torch.save({"step": step, "state_dict": model.state_dict(), "critic": critic.state_dict(),
                    "g_opt": g_opt.state_dict(), "d_opt": d_opt.state_dict()}, last + ".tmp")
        os.replace(last + ".tmp", last)

    def run_eval():
        ev = evaluate(model, val_loader, a.device, a.max_val_batches)
        log({"eval": dict(ev, step=step)})
        if vocoder:
            write_samples(model, val_ds, picked, vocoder, os.path.join(a.out, "samples", f"step_{step:06d}"),
                          a.device, ceiling_dir=os.path.join(a.out, "samples", "ceiling") if step == 0 else None)
        hook()

    set_mode(model)
    if step == 0:
        run_eval()

    seg_gen = torch.Generator().manual_seed(a.seed + 1)
    t0, acc = time.time(), {"d": 0.0, "adv": 0.0, "fm": 0.0, "sup": 0.0, "n": 0}
    while step < a.steps:
        for batch in train_loader:
            if step >= a.steps:
                break
            b = batch_to(batch, a.device)
            with torch.no_grad():                                   # frozen encoder, real durations
                enc, _ = model.encode(b["tokens"], b["voices"], b["token_lengths"],
                                      b.get("lang_ids") if model.langs is not None else None)
                expanded = model.expand(enc, b["durations"], b["frame_lengths"])
            pred = model.decode(expanded)
            fake_seg, real_seg = random_segments(pred, b["features"], b["frame_lengths"], a.segment, seg_gen)

            # critic step
            d_opt.zero_grad(set_to_none=True)
            d_loss = d_loss_fn(critic(real_seg, b["voices"]), critic(fake_seg.detach(), b["voices"]))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), a.grad_clip)
            d_opt.step()

            # generator (decoder) step
            g_opt.zero_grad(set_to_none=True)
            sup = masked_feature_loss(pred, b["features"], b["frame_lengths"])
            with torch.no_grad():
                real_outs = critic(real_seg, b["voices"])
            adv, fm = g_losses(real_outs, critic(fake_seg, b["voices"]))
            (sup + a.adv_weight * adv + a.fm_weight * fm).backward()
            torch.nn.utils.clip_grad_norm_(g_params, a.grad_clip)
            g_opt.step()
            step += 1

            for k, v in (("d", d_loss), ("adv", adv), ("fm", fm), ("sup", sup)):
                acc[k] += float(v.detach()) if torch.is_tensor(v) else float(v)
            acc["n"] += 1
            if step % a.log_every == 0:
                n = acc.pop("n")
                log(dict({"step": step}, **{k: v / n for k, v in acc.items()},
                         steps_per_s=n / (time.time() - t0)))
                t0, acc = time.time(), {"d": 0.0, "adv": 0.0, "fm": 0.0, "sup": 0.0, "n": 0}
            if step % a.eval_every == 0 or step == a.steps:
                checkpoint(final=step == a.steps)
                run_eval()

    # the report verifies the frozen part is untouched; so do we
    now = model.state_dict()
    changed = [k for k, v in frozen_ref.items() if not torch.equal(now[k].cpu(), v)]
    msg = (f"frozen check: {len(frozen_ref) - len(changed)}/{len(frozen_ref)} tensors unchanged"
           + (f" -- CHANGED: {changed}" if changed else ""))
    print(msg, flush=True)
    if changed:
        raise RuntimeError(msg)
    hook()
    return {"steps": step, "frozen_unchanged": len(frozen_ref), "out": a.out}


if __name__ == "__main__":
    main()
