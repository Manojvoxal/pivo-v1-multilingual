// Web Worker: loads the model once, turns text into audio and streams it to the page.
//
// page -> worker  { type: "load" } | { type: "synth", id, text, speed, voice, lang } | { type: "cancel", id }
//                 | { type: "prefetch" } (load the English dictionary now)
// worker -> page  progress / ready / dictionary / text / audio / done / cancelled / error
import * as ort from "./vendor/ort.wasm.min.mjs";
import { PiVoEngine, MODEL_FILES, SAMPLE_RATE } from "./engine.js";
import * as frontend from "./frontend.js";
import * as multi from "./frontend_multi.js";

const MAX_TOKENS = 160;      // tokens per utterance before splitting (about 10-15 s of speech)
const GAP_SENTENCE = 0.18;   // seconds of silence between sentences
const GAP_CLAUSE = 0.08;     // ... and between pieces of one long sentence

const here = (path) => new URL(path, import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = here("./vendor/");

let engine = null;
let manifest = null;
let lexiconPromise = null;
const cancelled = new Set();
let queue = Promise.resolve();

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url.split("/").pop()} (HTTP ${res.status}).`);
  if (!res.body) { const buf = await res.arrayBuffer(); onBytes(buf.byteLength); return buf; }
  const reader = res.body.getReader();
  const parts = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    size += value.byteLength;
    onBytes(value.byteLength);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.byteLength; }
  return out.buffer;
}

async function load() {
  const t0 = performance.now();
  const res = await fetch(here("./manifest.json"), { cache: "no-cache" });
  if (!res.ok) throw new Error(`Could not download manifest.json (HTTP ${res.status}).`);
  manifest = await res.json();
  const asset = (key) => here(`./${manifest.files[key].path}?v=${manifest.files[key].hash}`);
  const files = ["wasm", ...MODEL_FILES].map((key) => ({ key, url: asset(key) }));
  const total = files.reduce((a, f) => a + manifest.files[f.key].size, 0);
  let received = 0, lastPost = 0;
  const onBytes = (n) => {
    received += n;
    const now = performance.now();
    if (now - lastPost > 80) { lastPost = now; post({ type: "progress", stage: "download", fraction: Math.min(received / total, 1) }); }
  };
  const buffers = Object.fromEntries(await Promise.all(files.map(async (f) => [f.key, await fetchWithProgress(f.url, onBytes)])));
  post({ type: "progress", stage: "prepare", fraction: 1 });
  ort.env.wasm.wasmBinary = buffers.wasm;
  engine = await PiVoEngine.create(ort, buffers, manifest.config || {});
  // warm-up: the first run of each graph is slower, so pay that cost before the user clicks
  const warmVoice = manifest.voices.find((v) => v.language === "kn") || manifest.voices[0];
  const warm = multi.phonemize("ನಮಸ್ಕಾರ", "kn").ids;
  for await (const _ of engine.stream(warm, warmVoice.index, 1.0, { firstBlock: 4, block: 4, lang: warmVoice.lang_id })) break;
  post({ type: "ready", seconds: (performance.now() - t0) / 1000, voices: manifest.voices });
  // fetch the English dictionary in the background so code-mixed text is instant later
  setTimeout(() => ensureLexicon().catch(() => {}), 1500);
}

function ensureLexicon() {
  if (frontend.lexiconLoaded()) return Promise.resolve();
  if (!lexiconPromise) {
    lexiconPromise = (async () => {
      const lex = manifest.files.lexicon;
      const res = await fetch(here(`./${lex.path}?v=${lex.hash}`));
      if (!res.ok) throw new Error("Could not download the English dictionary.");
      frontend.setLexicon(await res.text());
      post({ type: "dictionary" });
    })().catch((e) => { lexiconPromise = null; throw e; });
  }
  return lexiconPromise;
}

const findVoice = (id) => manifest.voices.find((v) => v.id === id) || manifest.voices[0];

async function synth({ id, text, speed, voice, lang }) {
  if (cancelled.has(id)) return post({ type: "cancelled", id });
  const t0 = performance.now();
  const v = findVoice(voice);
  lang = lang || v.language;
  const langId = manifest.languages?.[lang] ?? v.lang_id ?? 0;
  if (multi.lexiconNeeded(text, lang) && !frontend.lexiconLoaded()) {
    post({ type: "status", id, message: "Loading the English dictionary…" });
    await ensureLexicon();
  }
  const front = multi.phonemize(text, lang);
  const chunks = frontend.splitUtterances(front.ids, MAX_TOKENS);
  const phones = chunks.reduce((n, c) => n + c.filter((t) => t >= 10).length, 0);
  const estimate = (phones * 0.075) / speed + (chunks.length - 1) * GAP_SENTENCE;
  post({ type: "text", id, lang, normalized: front.normalized, skipped: [...new Set(front.skipped)],
    unknown: [...new Set(front.unknown)], chunks: chunks.length, estimate });
  if (!chunks.length) return post({ type: "done", id, empty: true });

  let samples = 0, firstAudio = null;
  const stop = () => cancelled.has(id);
  for (let c = 0; c < chunks.length; c++) {
    if (c > 0) {
      const prev = chunks[c - 1];
      const gap = [4, 6, 7].includes(prev[prev.length - 2]) ? GAP_SENTENCE : GAP_CLAUSE;
      const silence = new Float32Array(Math.round(gap * SAMPLE_RATE));
      samples += silence.length;
      post({ type: "audio", id, pcm: silence }, [silence.buffer]);
    }
    for await (const pcm of engine.stream(chunks[c], v.index, speed, { stop, lang: langId })) {
      if (firstAudio === null) firstAudio = performance.now() - t0;
      samples += pcm.length;
      post({ type: "audio", id, pcm }, [pcm.buffer]);
    }
    if (stop()) { cancelled.delete(id); return post({ type: "cancelled", id }); }
    post({ type: "progress", id, stage: "generate", fraction: (c + 1) / chunks.length });
  }
  const seconds = (performance.now() - t0) / 1000;
  post({ type: "done", id, audioSeconds: samples / SAMPLE_RATE, seconds, firstAudio: firstAudio / 1000 });
}

self.onmessage = ({ data }) => {
  if (data.type === "cancel") { cancelled.add(data.id); return; }
  if (data.type === "prefetch") { if (manifest) ensureLexicon().catch(() => {}); return; }
  if (data.type === "load") {
    load().catch((e) => post({ type: "error", stage: "load", message: e.message || String(e) }));
    return;
  }
  if (data.type === "synth") {
    // one generation at a time; a new request waits for (or cancels) the previous one
    queue = queue.then(() => synth(data)).catch((e) => post({ type: "error", id: data.id, message: e.message || String(e) }));
  }
};
