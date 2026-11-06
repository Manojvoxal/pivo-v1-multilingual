"""Export a trained acoustic checkpoint to PiVo's ONNX runtime format.

    PiVo-KN (Kannada + English):
    python export_onnx.py --ckpt runs/kn_en_v1_refine/refined.pt \
        --pivo-dir ../models/PiVo-1.4M --out ../models/PiVo-KN-1.4M

    Multilingual (checkpoint kind pivo-acoustic-v1-multi*, 12 languages, 24 voices): the encoder
    gets a fourth input `lang` (int64 [batch], the language id of the text, see metadata
    "language_ids"); voices are named <lang>_<f|m>. On Modal: `modal run train_gpu.py --stage export`.
    python export_onnx.py --ckpt runs/multi_w128_refine/refined.pt --pivo-dir ../models/PiVo-1.4M \
        --cmudict ../src/pivo/resources/cmudict.dict --out ../models/PiVo-Multi-1.8M

Writes a model folder with the same layout and contract as PiVo-1.4M, so the existing
runtime (src/pivo) can load it:

  onnx/encoder.onnx       new: tokens, voice, lengths -> encoded, log_durations
  onnx/decoder.onnx       new: streaming decoder with the 63-frame attention cache
  onnx/conditioning.onnx  copied unchanged from PiVo-1.4M (the vocoder was not retrained)
  onnx/vocoder_step.onnx  copied unchanged from PiVo-1.4M
  onnx/metadata.json      127 symbols, 4 voices, frontend "pivo-kn-en-v1", parity numbers
  model_manifest.json     sha256 + size of the five runtime files (checked by the runtime)

Before writing anything final it checks the exported graphs against the PyTorch model
(full-sequence and streaming decode at chunk sizes 1, 16 and 47, all voices, English and
Kannada token sequences) and refuses to finish if the difference exceeds --tol.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data_pipeline"))
sys.path.insert(0, os.path.join(HERE, "data_pipeline_multi"))
from pivo_train.model import PiVoAcoustic, phase_features  # noqa: E402

CACHE_NAMES = ["key0", "value0", "key1", "value1"]
STEP_STATE_NAMES = ["memory", "gru1", "gru2", "gru3", "frame_state"]
DEFAULT_VOICE_NAMES = {"en_hifi92_f": "en_f", "en_hifi6097_m": "en_m",
                       "kn_syspin_f": "kn_f", "kn_syspin_m": "kn_m"}
TEST_TEXT = [
    ("en", "NASA is ready for the next mission."),
    ("en", "Hello, my name is Vo. How many parameters does a voice need?"),
    ("kn", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?"),
    ("kn", "ಸ್ವಲ್ಪ ಬಾಯಿ ಉಗ್ಗಾದರೆ ಏನಂತೆ? ದುಂಡುಮುಖ, ಬಿಳಿ ಮೈ, ಕಣ್ಣು ಮೂಗು ಬಾಯಿ ಕಿವಿ ಎಲ್ಲಾ ನೆಟ್ಟಗಿಲ್ಲೇನು?"),
    ("kn", "ನನ್ನ phone ಎಲ್ಲಿದೆ ಎಂದು ಹೇಳಿ."),
]
# multilingual parity sentences (one per language; all pass the strict training frontend)
MULTI_TEST_TEXT = {
    "en": "Hello, my name is Vo. How many parameters does a voice need?",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?",
    "hi": "नमस्ते, आप कैसे हैं?",
    "te": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "bn": "নমস্কার, আপনি কেমন আছেন?",
    "mr": "नमस्कार, तुम्ही कसे आहात?",
    "gu": "નમસ્તે, તમે કેમ છો?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "ml": "നമസ്കാരം, സുഖമാണോ?",
    "or": "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "as": "নমস্কাৰ, আপুনি কেনে আছে?",
}


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, voice, lengths):
        return self.model.encode(tokens, voice, lengths)


class MultiEncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, voice, lang, lengths):
        return self.model.encode(tokens, voice, lengths, lang=lang)


class DecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, expanded, position, key0, value0, key1, value1):
        features, caches = self.model.decode_step(expanded, position, [key0, value0, key1, value1])
        return (features, *caches)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(model, onnx_dir):
    tokens = torch.tensor([[1, 21, 38, 25, 40, 3, 62, 32, 4, 2]])
    dyn_enc = {"tokens": {0: "batch", 1: "tokens"}, "voice": {0: "batch"}, "lengths": {0: "batch"},
               "encoded": {0: "batch", 1: "tokens"}, "log_durations": {0: "batch", 1: "tokens"}}
    if model.langs is not None:
        graph, args, names = (MultiEncoderGraph(model), (tokens, torch.tensor([0]), torch.tensor([0]),
                              torch.tensor([tokens.shape[1]])), ["tokens", "voice", "lang", "lengths"])
        dyn_enc["lang"] = {0: "batch"}
    else:
        graph, args, names = (EncoderGraph(model), (tokens, torch.tensor([0]), torch.tensor([tokens.shape[1]])),
                              ["tokens", "voice", "lengths"])
    torch.onnx.export(
        graph.eval(), args, os.path.join(onnx_dir, "encoder.onnx"), dynamo=False, opset_version=17,
        input_names=names, output_names=["encoded", "log_durations"], dynamic_axes=dyn_enc)
    c = model.config
    hd = c["width"] // c["heads"]
    caches = [torch.zeros(1, c["kv_heads"], 5, hd) for _ in CACHE_NAMES]
    dyn = {"expanded": {0: "batch", 1: "frames"}, "features": {0: "batch", 1: "frames"}}
    for n in CACHE_NAMES:
        dyn[n] = {0: "batch", 2: "history"}
        dyn[n + "_out"] = {0: "batch", 2: "new_history"}
    torch.onnx.export(
        DecoderGraph(model).eval(), (torch.zeros(1, 7, c["width"] + 4), torch.tensor(5), *caches),
        os.path.join(onnx_dir, "decoder.onnx"), dynamo=False, opset_version=17,
        input_names=["expanded", "position", *CACHE_NAMES],
        output_names=["features", *[n + "_out" for n in CACHE_NAMES]], dynamic_axes=dyn)
    model.eval()  # the exporter restores the wrapper's train/eval flag onto the model; force eval for checks


def check(model, onnx_dir, token_sets, voices, cases=None):
    """Max abs difference between ONNX and PyTorch for encoder, durations and features.
    cases: optional explicit [(token_ids, voice, lang)] (multilingual); default = every token
    set with every voice."""
    import onnxruntime as ort
    enc = ort.InferenceSession(os.path.join(onnx_dir, "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(onnx_dir, "decoder.onnx"), providers=["CPUExecutionProvider"])
    c = model.config
    hd = c["width"] // c["heads"]
    worst = {"encoder": 0.0, "log_durations": 0.0, "decoder": 0.0}
    if cases is None:
        cases = [(toks, voice, None) for toks in token_sets for voice in voices]
    for toks, voice, lang in cases:
        if True:  # (kept the old two-level indentation)
            tokens = np.array([toks], dtype=np.int64)
            feeds = {"tokens": tokens, "voice": np.array([voice], np.int64),
                     "lengths": np.array([len(toks)], np.int64)}
            if lang is not None:
                feeds["lang"] = np.array([lang], np.int64)
            o_enc, o_logd = enc.run(None, feeds)
            with torch.no_grad():
                t_enc, t_logd = model.encode(torch.from_numpy(tokens), torch.tensor([voice]),
                                             torch.tensor([len(toks)]),
                                             lang=None if lang is None else torch.tensor([lang]))
            worst["encoder"] = max(worst["encoder"], float(np.abs(t_enc.numpy() - o_enc).max()))
            worst["log_durations"] = max(worst["log_durations"], float(np.abs(t_logd.numpy() - o_logd).max()))
            dur = model.predict_durations(torch.from_numpy(o_logd), torch.from_numpy(tokens))[0]
            expanded = np.repeat(o_enc[0], dur.numpy(), axis=0)[None]
            expanded = np.concatenate((expanded, phase_features(dur).numpy()[None]), -1).astype(np.float32)
            with torch.no_grad():
                ref = model.denormalize(model.decode(torch.from_numpy(expanded))).numpy()
            for chunk in (1, 16, 47):
                cache = {n: np.zeros((1, c["kv_heads"], 0, hd), np.float32) for n in CACHE_NAMES}
                out = []
                for p in range(0, expanded.shape[1], chunk):
                    r = dec.run(None, {"expanded": expanded[:, p:p + chunk], "position": np.array(p, np.int64),
                                       **cache})
                    out.append(r[0])
                    cache = dict(zip(CACHE_NAMES, r[1:]))
                worst["decoder"] = max(worst["decoder"], float(np.abs(np.concatenate(out, 1) - ref).max()))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="refined.pt (or a stage-1 best.pt)")
    ap.add_argument("--pivo-dir", required=True, help="PiVo-1.4M folder (for the unchanged vocoder graphs)")
    ap.add_argument("--out", required=True, help="new model folder, e.g. models/PiVo-KN-1.4M")
    ap.add_argument("--name", default=None, help="default: PiVo-KN-1.4M, or PiVo-Multi-<size>M")
    ap.add_argument("--default-voice", default="kn_f")
    ap.add_argument("--cmudict", default=None, help="default: <pivo-dir>/pivo/resources/cmudict.dict")
    ap.add_argument("--tol", type=float, default=2e-4)
    a = ap.parse_args(argv)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = PiVoAcoustic(config=ck["config"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    multi = model.langs is not None
    cmudict = a.cmudict or os.path.join(a.pivo_dir, "pivo", "resources", "cmudict.dict")
    if multi:
        from pivo_multi.frontend import phonemize
        from pivo_multi import english
        from pivo_multi.langs import LANG, LANG_TO_ID
    else:
        from pivo_kn_v3.frontend import phonemize
        from pivo_kn_v3 import english
    english.set_cmudict_path(cmudict)

    symbols = list(ck["symbols"])
    voices_ck = {int(k): tuple(v) for k, v in ck["voices"].items()}
    if multi:  # en_iitm_f -> en_f, kn_syspin_m -> kn_m, hi_rasa_f -> hi_f
        short = lambda spk, lang: f"{lang}_{spk.rsplit('_', 1)[-1]}"
        for vid, (spk, lang) in voices_ck.items():
            if vid // 2 != LANG_TO_ID[lang]:
                raise SystemExit(f"voice {vid} ({spk}) does not sit at 2*lang_id of {lang}")
    else:
        short = lambda spk, lang: DEFAULT_VOICE_NAMES.get(spk, spk)
    names = {short(spk, lang): vid for vid, (spk, lang) in sorted(voices_ck.items())}
    voice_info = {short(spk, lang): {"id": vid, "speaker": spk, "language": lang,
                                     **({"lang_id": LANG_TO_ID[lang]} if multi else {})}
                  for vid, (spk, lang) in sorted(voices_ck.items())}
    if a.default_voice not in names:
        raise SystemExit(f"--default-voice must be one of {sorted(names)}")

    onnx_dir = os.path.join(a.out, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    export(model, onnx_dir)
    for n in ("conditioning.onnx", "vocoder_step.onnx"):
        shutil.copyfile(os.path.join(a.pivo_dir, "onnx", n), os.path.join(onnx_dir, n))

    token_sets, cases = [], None
    texts = list(TEST_TEXT) if not multi else list(MULTI_TEST_TEXT.items())
    toks_by_lang = {}
    for lang, text in texts:
        front = phonemize(text, lang)
        if [symbols[i] for i in front.token_ids] != list(front.symbols):
            raise SystemExit("checkpoint symbol table does not match the frontend")
        token_sets.append(list(front.token_ids))
        toks_by_lang[lang] = list(front.token_ids)
    if multi:  # every voice: its own language's sentence + the English one
        cases = []
        for vid, (spk, lang) in sorted(voices_ck.items()):
            cases.append((toks_by_lang[lang], vid, LANG_TO_ID[lang]))
            if lang != "en":
                cases.append((toks_by_lang["en"], vid, LANG_TO_ID["en"]))
    worst = check(model, onnx_dir, token_sets, sorted(voices_ck), cases)
    print("ONNX vs PyTorch max abs diff:", {k: f"{v:.2e}" for k, v in worst.items()})
    if max(worst.values()) > a.tol:
        raise SystemExit(f"PARITY FAILED (tol {a.tol})")

    with open(os.path.join(a.pivo_dir, "onnx", "metadata.json")) as fh:
        base = json.load(fh)
    acoustic = sum(p.numel() for p in model.parameters())
    vocoder = base["parameters"]["components"]["vocoder"]["parameters"]
    if a.name is None:
        a.name = f"PiVo-Multi-{(acoustic + vocoder) / 1e6:.1f}M" if multi else "PiVo-KN-1.4M"
    meta = dict(base)
    # the acoustic model's own sizes (width 128 has head_dim 32, not PiVo's 24); vocoder keys stay
    meta["config"] = dict(base["config"], **{k: v for k, v in model.config.items() if k != "dropout"})
    meta["config"]["head_dim"] = model.config["width"] // model.config["heads"]
    langs_used = sorted({lang for _, lang in voices_ck.values()}, key=lambda c: LANG_TO_ID[c]) if multi else ["en", "kn"]
    meta.update({
        "symbols": symbols,
        "voices": names,
        "voice_info": voice_info,
        "default_voice": a.default_voice,
        "languages": langs_used,
        "frontend": "pivo-multi-v1" if multi else "pivo-kn-en-v1",
        **({"language_ids": {c: LANG_TO_ID[c] for c in langs_used},
            "language_names": {c: LANG[c]["name"] or "English" for c in langs_used},
            "encoder_inputs": ["tokens", "voice", "lang", "lengths"]} if multi else {}),
        "source_checkpoint": {"file": os.path.basename(a.ckpt), "sha256": sha256(a.ckpt),
                              "kind": ck.get("kind"), "step": ck.get("step")},
    })
    meta["parameters"] = dict(base["parameters"], provisional_name=a.name, total_parameters=acoustic + vocoder,
                              within_1_5m=acoustic + vocoder <= 1_500_000)
    meta["parameters"]["components"] = dict(base["parameters"]["components"], acoustic={
        "parameters": acoustic, "trainable": acoustic, "fp32_bytes": 4 * acoustic})
    meta["verification"] = dict(base["verification"])
    for n, key in (("encoder", "encoder"), ("decoder", "decoder")):
        meta["verification"][n] = {"max_absolute_error": max(worst[key], worst["log_durations"]) if n == "encoder"
                                   else worst[key], "bytes": os.path.getsize(os.path.join(onnx_dir, n + ".onnx"))}
    with open(os.path.join(onnx_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    files = []
    for n in sorted(os.listdir(onnx_dir)):
        p = os.path.join(onnx_dir, n)
        files.append({"path": f"onnx/{n}", "bytes": os.path.getsize(p), "sha256": sha256(p)})
    manifest = {"format": "pivo-model-manifest-v1", "model_name": a.name,
                "total_inference_parameters": acoustic + vocoder, "sample_rate": base["config"]["sample_rate"],
                "files": files}
    with open(os.path.join(a.out, "model_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    for extra in ("LICENSE", "NOTICE", "THIRD_PARTY.md"):
        if os.path.exists(os.path.join(a.pivo_dir, extra)):
            shutil.copyfile(os.path.join(a.pivo_dir, extra), os.path.join(a.out, extra))
    if os.path.isdir(os.path.join(a.pivo_dir, "LICENSES")):
        shutil.copytree(os.path.join(a.pivo_dir, "LICENSES"), os.path.join(a.out, "LICENSES"), dirs_exist_ok=True)
    print(f"wrote {a.out}: {acoustic:,} acoustic + {vocoder:,} vocoder = {acoustic + vocoder:,} params, "
          f"voices {names}")


if __name__ == "__main__":
    main()
