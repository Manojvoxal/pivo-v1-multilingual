"""Find the speakers inside an unlabeled multi-speaker corpus (SPRINGLab/IndicTTS-English
has 151k Indian-English clips from ~26 IIT Madras voice talents, but no speaker column).

1. per clip: duration, median F0 (gender) and a speaker embedding (resemblyzer GE2E, 256-d)
   of the first 3 s
2. if the stored file names carry the speaker (e.g. train_tamilfemale_00001.wav), group by
   name; otherwise cluster the embeddings (k-means, k picked by silhouette)
3. per speaker: hours, gender, which parquet shards it lives in, sample rows
"""

import collections
import re

import numpy as np

SR = 16000


def median_f0(audio, sr=SR, fmin=70.0, fmax=400.0, frame=640, hop=320):
    """Median pitch (Hz) of voiced frames via normalised autocorrelation; 0 if unvoiced."""
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < frame:
        return 0.0
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(frame)[None, :]
    frames -= frames.mean(1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(1))
    keep = rms > 0.1 * rms.max() if rms.max() > 0 else np.zeros(n, bool)
    if not keep.any():
        return 0.0
    frames = frames[keep]
    spec = np.fft.rfft(frames, 2 * frame)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :frame]
    ac /= ac[:, :1] + 1e-9
    lo, hi = int(sr / fmax), int(sr / fmin)
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    peaks = ac[np.arange(len(ac)), lags]
    voiced = peaks > 0.45
    if voiced.sum() < 5:
        return 0.0
    return float(np.median(sr / lags[voiced]))


def path_speaker_keys(paths, min_rows=300, min_speakers=6, coverage=0.95):
    """Speaker key from file names (stem without the trailing number), or None when the
    names do not look like per-speaker names."""
    keys = []
    for p in paths:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(p or "").split("/")[-1].split("\\")[-1])
        k = re.sub(r"[_\-.]*\d+$", "", stem).lower()
        keys.append(k or None)
    counts = collections.Counter(k for k in keys if k)
    big = {k for k, c in counts.items() if c >= min_rows}
    covered = sum(counts[k] for k in big) / max(1, len(keys))
    if len(big) < min_speakers or covered < coverage:
        return None
    return [k if k in big else None for k in keys]


def cluster_embeddings(emb, k_range=(6, 40), sample=6000, seed=0, merge_cos=0.92):
    """k-means on L2-normalised embeddings; k chosen by the cosine silhouette on a sample;
    clusters whose centroids are nearly identical are merged. Returns (labels, info)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    sub = x[rng.choice(len(x), size=min(sample, len(x)), replace=False)]
    best = (-1.0, None, None)
    scores = {}
    for k in range(k_range[0], min(k_range[1], len(sub) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(sub)
        s = silhouette_score(sub, km.labels_, metric="cosine",
                             sample_size=min(3000, len(sub)), random_state=seed)
        scores[k] = round(float(s), 4)
        if s > best[0]:
            best = (s, k, km)
    _, k, km = best
    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sims = cent @ cent.T
    for a in range(k):
        for b in range(a + 1, k):
            if sims[a, b] > merge_cos:
                parent[find(a)] = find(b)
    raw = np.argmax(x @ cent.T, axis=1)
    roots = sorted({find(c) for c in range(k)})
    remap = {r: i for i, r in enumerate(roots)}
    labels = np.array([remap[find(c)] for c in raw])
    return labels, {"k_best": k, "silhouette": scores, "merged_to": len(roots)}


def summarize(speaker_ids, durations, f0s, shards, female_f0=165.0):
    """Per speaker: clips, hours, median F0, gender guess, shards it appears in."""
    out = {}
    by = collections.defaultdict(list)
    for i, s in enumerate(speaker_ids):
        if s is not None:
            by[s].append(i)
    for s, idx in by.items():
        f0 = [f0s[i] for i in idx if f0s[i] > 0]
        med = float(np.median(f0)) if f0 else 0.0
        sh = collections.Counter(shards[i] for i in idx)
        out[str(s)] = {
            "clips": len(idx), "hours": round(sum(durations[i] for i in idx) / 3600, 2),
            "median_f0": round(med, 1), "gender": "f" if med >= female_f0 else "m",
            "shards": dict(sh.most_common(6)), "n_shards": len(sh),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["hours"]))


def pick_voices(summary, explicit=None, max_single_hours=16.0):
    """Default: the female and the male speaker with the most hours. A cluster above
    `max_single_hours` is probably two talents merged (IndicTTS English has at most
    ~13 h per talent), so it is only used when nothing else of that gender exists."""
    if explicit:
        return {summary[s]["gender"]: s for s in explicit}
    picks = {}
    for s, info in summary.items():  # sorted by hours, largest first
        if info["hours"] <= max_single_hours:
            picks.setdefault(info["gender"], s)
    for s, info in summary.items():
        picks.setdefault(info["gender"], s)
    return picks
