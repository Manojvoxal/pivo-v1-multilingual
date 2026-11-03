"""Stage-1 training (supervised) for the PiVo acoustic model.

Two modes:
  * --from-scratch (multilingual model, pivo-v1-multilingual): a fresh model of --width,
    vocab / voices / languages sized from the dataset (symbols.json, max voice_id / lang_id),
    feature mean/std from manifests/feature_stats.json, language embedding on.
  * default (PiVo-KN): warm-start from PiVo's English acoustic.pt, grow the voice table and
    initialise Kannada token rows from similar English phones.

Then trains on the manifests produced by the data pipeline:

    loss = SmoothL1(normalized features) + 0.1 * MSE(log durations, log(1 + frames))

(the stage-1 recipe from the PiVo technical report). Writes checkpoints, a JSONL
training log and, every --eval-every steps, validation losses plus one WAV per voice
synthesised end-to-end (predicted durations -> features -> PiVo's FARGAN vocoder).

Run locally:  python -m pivo_train.train --data-root ... --pivo-ckpt .../acoustic.pt --onnx-dir .../onnx --out runs/x
On Modal:     see train_modal.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BucketBatchSampler, PiVoDataset, collate, describe, read_manifest
from .model import load_pivo_checkpoint, new_model, PiVoAcoustic  # noqa: F401


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="dataset root containing manifests/")
    p.add_argument("--pivo-ckpt", default=None, help="PiVo-1.4M checkpoints/acoustic.pt (warm start)")
    p.add_argument("--from-scratch", action="store_true", help="multilingual: fresh model, no PiVo weights")
    p.add_argument("--width", type=int, default=96, help="model width for --from-scratch (96, 112, 128...)")
    p.add_argument("--onnx-dir", default=None, help="PiVo-1.4M onnx/ dir (vocoder for sample WAVs)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-frames", type=int, default=16 * 1200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--duration-weight", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-val-batches", type=int, default=50)
    p.add_argument("--amp", choices=["none", "bf16"], default="none", help="mixed precision on GPU")
    p.add_argument("--preload", action="store_true", help="load all feature files into RAM first")
    p.add_argument("--pack-dir", default=None, help="where to keep a one-file feature cache (built once)")
    p.add_argument("--resume", default=None, help="checkpoint to resume from ('auto' = <out>/last.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def masked_losses(pred, target, frame_lengths, log_dur, durations, token_lengths):
    fmask = (torch.arange(pred.shape[1], device=pred.device)[None] < frame_lengths[:, None]).float()
    feat = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    feat = (feat * fmask).sum() / fmask.sum()
    tmask = (torch.arange(log_dur.shape[1], device=pred.device)[None] < token_lengths[:, None]).float()
    dur = (log_dur - torch.log1p(durations.float())).pow(2)
    dur = (dur * tmask).sum() / tmask.sum()
    return feat, dur


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))


def voice_map(rows):
    """voice_id -> (speaker, lang); PiVo source voice for warm start: female -> Pi(0), male -> Vo(1)."""
    voices = {}
    for r in rows:
        voices[r["voice_id"]] = (r["speaker"], r["lang"])
    n = max(voices) + 1
    init = [1 if voices.get(v, ("", ""))[0].endswith("_m") else 0 for v in range(n)]
    return voices, init


def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, args, max_batches):
    model.eval()
    tot_f = tot_d = n = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = batch_to(batch, args.device)
        pred, logd = model(b["tokens"], b["voices"], b["token_lengths"], b["durations"], b["frame_lengths"],
                           lang=b.get("lang_ids") if model.langs is not None else None)
        f, d = masked_losses(pred, b["features"], b["frame_lengths"], logd, b["durations"], b["token_lengths"])
        tot_f += f.item()
        tot_d += d.item()
        n += 1
    model.train()
    return {"val_feature": tot_f / max(n, 1), "val_duration": tot_d / max(n, 1)}


@torch.no_grad()
def write_samples(model, rows, vocoder, out_dir, device):
    """One validation sentence per voice, synthesised with *predicted* durations."""
    from .vocoder import write_wav
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    seen = set()
    for r in rows:
        if r["voice_id"] in seen:
            continue
        seen.add(r["voice_id"])
        tokens = torch.tensor(r["token_ids"], device=device)
        feats, _ = model.synthesize_features(tokens, r["voice_id"],
                                             lang=r.get("lang_id") if model.langs is not None else None)
        audio = vocoder(feats.cpu().numpy())
        name = f"voice{r['voice_id']:02d}_{r['speaker']}"
        write_wav(os.path.join(out_dir, name + ".wav"), audio)
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(r["text"] + "\n")
    model.train()


def save(path, model, opt, step, args, extra):
    tmp = path + ".tmp"
    torch.save({
        "kind": "pivo-acoustic-v1-multi" if model.langs is not None else "pivo-acoustic-v1-kn",
        "step": step, "config": model.config,
        "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args), **extra,
    }, tmp)
    os.replace(tmp, path)


def main(argv=None, on_checkpoint=None):
    """on_checkpoint: optional callback run after every save / sample write (e.g. Modal volume commit)."""
    args = parse_args(argv)
    hook = on_checkpoint or (lambda: None)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")

    train_rows = read_manifest(args.data_root, "train")
    val_rows = []
    for split in ("val", "test"):  # tiny/smoke datasets may lack a val split
        if os.path.exists(os.path.join(args.data_root, "manifests", f"{split}.jsonl")):
            val_rows = read_manifest(args.data_root, split)
            break
    if not val_rows:
        val_rows, train_rows = train_rows[:max(1, len(train_rows) // 50)], train_rows[max(1, len(train_rows) // 50):]
    with open(os.path.join(args.data_root, "manifests", "symbols.json"), encoding="utf-8") as fh:
        symbols = json.load(fh)["symbols"]
    voices, voice_init = voice_map(train_rows + val_rows)
    print("train:", json.dumps(describe(train_rows), indent=1))
    print("val clips:", len(val_rows), "| voices:", voices,
          "" if args.from_scratch else f"| warm-start voice rows from PiVo: {voice_init}")

    if args.from_scratch:
        with open(os.path.join(args.data_root, "manifests", "feature_stats.json")) as fh:
            fstats = json.load(fh)
        all_rows = train_rows + val_rows
        n_langs = max(int(r.get("lang_id", 0)) for r in all_rows) + 1
        vocab = max(len(symbols), json.load(open(os.path.join(args.data_root, "manifests", "symbols.json"),
                                                 encoding="utf-8")).get("vocab_size", len(symbols)))
        model = new_model(width=args.width, vocab_size=vocab, num_voices=len(voice_init), num_langs=n_langs,
                          feature_mean=fstats["mean"], feature_std=fstats["std"], dropout=args.dropout)
        pivo = {"kind": "from-scratch"}
        print(f"from scratch: width {args.width}, vocab {vocab}, voices {len(voice_init)}, languages {n_langs}, "
              f"{sum(p.numel() for p in model.parameters()):,} acoustic params", flush=True)
    else:
        if not args.pivo_ckpt:
            raise SystemExit("--pivo-ckpt is required unless --from-scratch")
        model, pivo = load_pivo_checkpoint(args.pivo_ckpt, num_voices=len(voice_init),
                                           voice_init=voice_init, symbols=symbols)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout
    model.to(args.device).train()
    mean, std = model.feature_mean.cpu().numpy(), model.feature_std.cpu().numpy()

    t_load = time.time()
    data_tag = os.path.basename(os.path.normpath(args.data_root))
    pack = lambda split: os.path.join(args.pack_dir, f"{data_tag}_{split}") if args.pack_dir else None
    train_ds = PiVoDataset(args.data_root, train_rows, mean, std, max_frames=model.config["max_utterance_frames"],
                           preload=args.preload, pack_path=pack("train"))
    val_ds = PiVoDataset(args.data_root, val_rows, mean, std, preload=args.preload, pack_path=pack("val"))
    if args.preload:
        print(f"preloaded {len(train_ds) + len(val_ds)} feature files in {time.time() - t_load:.0f}s", flush=True)
        hook()  # persist the pack cache
    sampler = BucketBatchSampler(train_ds.rows, args.batch_size, args.max_batch_frames, seed=args.seed)
    loader_kw = dict(collate_fn=collate, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=min(2, args.num_workers))

    emb = ("voices", "tokens", "langs")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not any(e in n for e in emb)]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not any(e in n for e in emb))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

    step, epoch, best = 0, 0, float("inf")
    resume = os.path.join(args.out, "last.pt") if args.resume == "auto" else args.resume
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        step, epoch, best = ck["step"], ck.get("epoch", 0), ck.get("best_val", best)
        print(f"resumed from {resume} at step {step}")

    vocoder = None
    if args.onnx_dir:
        from .vocoder import Vocoder
        vocoder = Vocoder(args.onnx_dir)
    extra = lambda: {"symbols": symbols, "voices": {str(k): v for k, v in voices.items()},
                     "voice_init": voice_init, "epoch": epoch, "best_val": best,
                     "pivo_parent": {"kind": pivo.get("kind"), "trained_steps": pivo.get("trained_steps")}}

    if step == 0:
        ev = evaluate(model, val_loader, args, args.max_val_batches)
        print("before training:", ev, flush=True)
        if vocoder:
            write_samples(model, val_ds.rows, vocoder, os.path.join(args.out, "samples", "step_000000"), args.device)
        hook()

    t0, run_f, run_d, run_n = time.time(), 0.0, 0.0, 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
            b = batch_to(batch, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16" and args.device.startswith("cuda")):
                pred, logd = model(b["tokens"], b["voices"], b["token_lengths"], b["durations"], b["frame_lengths"],
                                   lang=b.get("lang_ids") if model.langs is not None else None)
            f, d = masked_losses(pred.float(), b["features"], b["frame_lengths"], logd.float(), b["durations"], b["token_lengths"])
            loss = f + args.duration_weight * d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            run_f, run_d, run_n = run_f + f.item(), run_d + d.item(), run_n + 1

            if step % args.log_every == 0:
                rec = {"step": step, "epoch": epoch, "feature": run_f / run_n, "duration": run_d / run_n,
                       "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
                       "steps_per_s": run_n / (time.time() - t0)}
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                t0, run_f, run_d, run_n = time.time(), 0.0, 0.0, 0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_loader, args, args.max_val_batches)
                ev["step"] = step
                print("eval:", json.dumps(ev), flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"eval": ev}) + "\n")
                score = ev["val_feature"] + args.duration_weight * ev["val_duration"]
                if score < best:
                    best = score
                    save(os.path.join(args.out, "best.pt"), model, opt, step, args, extra())
                if vocoder:
                    write_samples(model, val_ds.rows, vocoder,
                                  os.path.join(args.out, "samples", f"step_{step:06d}"), args.device)
                hook()

            if step % args.save_every == 0 or step == args.steps:
                save(os.path.join(args.out, "last.pt"), model, opt, step, args, extra())
                if step % args.eval_every == 0 or step == args.steps:  # keep numbered copies sparse
                    save(os.path.join(args.out, f"step_{step:06d}.pt"), model, opt, step, args, extra())
                hook()
        epoch += 1
    hook()
    print(f"done: {step} steps, best val {best:.4f}", flush=True)
    return {"steps": step, "best_val": best}


if __name__ == "__main__":
    main()
