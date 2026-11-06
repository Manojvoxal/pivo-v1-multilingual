// PiVo browser runtime: the same steps as src/pivo/runtime.py, on onnxruntime-web (WebAssembly).
// It runs inside a Web Worker (worker.js), so the page never freezes while audio is generated.
// Checked against the Python runtime: identical token ids, audio within 1e-4.
// Multilingual models (PiVo-Multi) have a fourth encoder input `lang`; the attention head size
// depends on the model width (96 -> 24, 128 -> 32) and comes from the model metadata.

const CFG = { maxTokenFrames: 200, maxUtteranceFrames: 3000, kvHeads: 2, headDim: 24, chunk: 16, numSpecial: 10, space: 3 };
const STATE = [["memory", 256], ["gru1", 160], ["gru2", 128], ["gru3", 128], ["frame_state", 164]];
const CACHE = ["key0", "value0", "key1", "value1"];
export const MODEL_FILES = ["encoder", "decoder", "conditioning", "vocoder_step"];
export const SAMPLE_RATE = 16000;
export const FRAME = 160;

// Yield to the worker's event loop (so a "cancel" message can arrive) without timer clamping.
const channel = new MessageChannel();
const waiting = [];
channel.port1.onmessage = () => waiting.shift()?.();
const tick = () => new Promise((resolve) => { waiting.push(resolve); channel.port2.postMessage(0); });

export class PiVoEngine {
  /** buffers: { encoder: ArrayBuffer, ... }; config: { head_dim, kv_heads } from the model metadata */
  static async create(ort, buffers, config = {}) {
    const e = new PiVoEngine();
    e.ort = ort;
    e.s = {};
    e.cfg = { ...CFG, headDim: config.head_dim ?? CFG.headDim, kvHeads: config.kv_heads ?? CFG.kvHeads };
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    for (const name of MODEL_FILES) e.s[name] = await ort.InferenceSession.create(new Uint8Array(buffers[name]), opts);
    e.hasLang = e.s.encoder.inputNames.includes("lang");
    return e;
  }

  i64(arr, dims) { return new this.ort.Tensor("int64", BigInt64Array.from(arr, (v) => BigInt(v)), dims); }
  f32(data, dims) { return new this.ort.Tensor("float32", data, dims); }

  /** Token ids -> durations + acoustic features [frames x 20]. `lang`: language id (multilingual models). */
  async features(ids, voice, speed = 1.0, lang = 0) {
    const CFG = this.cfg;
    const T = ids.length;
    const feeds = { tokens: this.i64(ids, [1, T]), voice: this.i64([voice], [1]), lengths: this.i64([T], [1]) };
    if (this.hasLang) feeds.lang = this.i64([lang], [1]);
    const enc = await this.s.encoder.run(feeds);
    const encoded = enc.encoded.data, logd = enc.log_durations.data, W = enc.encoded.dims[2];
    const lmax = Math.log1p(CFG.maxTokenFrames);
    const dur = ids.map((tok, i) => {
      if (tok <= 2) return 0;
      let d = Math.round(Math.expm1(Math.min(Math.max(logd[i], 0), lmax)) / speed);
      d = Math.min(Math.max(d, 0), CFG.maxTokenFrames);
      return tok >= CFG.numSpecial ? Math.max(d, 1) : d;
    });
    const F = dur.reduce((a, b) => a + b, 0);
    if (F > CFG.maxUtteranceFrames) return { tooLong: true, frames: F };
    const width = W + 4;
    const expanded = new Float32Array(F * width);
    let f = 0;
    for (let i = 0; i < T; i++) {
      const n = dur[i];
      for (let k = 0; k < n; k++, f++) {
        const row = expanded.subarray(f * width, (f + 1) * width);
        row.set(encoded.subarray(i * W, (i + 1) * W));
        const frac = (k + 0.5) / n;
        row[W] = 2 * frac - 1;
        row[W + 1] = Math.sin(Math.PI * frac);
        row[W + 2] = Math.cos(Math.PI * frac);
        row[W + 3] = Math.log1p(n) / lmax;
      }
    }
    let cache = Object.fromEntries(CACHE.map((n) => [n, this.f32(new Float32Array(0), [1, CFG.kvHeads, 0, CFG.headDim])]));
    const feats = new Float32Array(F * 20);
    for (let p = 0; p < F; p += CFG.chunk) {
      const n = Math.min(CFG.chunk, F - p);
      const r = await this.s.decoder.run({
        expanded: this.f32(expanded.slice(p * width, (p + n) * width), [1, n, width]),
        position: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(p)]), []),
        ...cache,
      });
      feats.set(r.features.data, p * 20);
      cache = Object.fromEntries(CACHE.map((c) => [c, r[c + "_out"]]));
    }
    return { feats, frames: F };
  }

  /**
   * Token ids -> PCM blocks (Float32Array, 16 kHz). Yields `firstBlock` frames first for a fast
   * start, then `block` frames at a time. `stop()` is polled between frames.
   */
  async *stream(ids, voice, speed, { firstBlock = 8, block = 20, stop = () => false, lang = 0 } = {}) {
    const f = await this.features(ids, voice, speed, lang);
    if (f.tooLong) {
      // over the 30 s utterance budget: split between words and do each half
      const spaces = ids.map((id, i) => (id === CFG.space ? i : -1)).filter((i) => i > 0);
      if (!spaces.length) throw new Error("This sentence is too long to read in one go. Add a full stop or comma.");
      const mid = spaces[Math.floor(spaces.length / 2)];
      yield* this.stream([...ids.slice(0, mid), 2], voice, speed, { firstBlock, block, stop, lang });
      yield* this.stream([1, ...ids.slice(mid + 1)], voice, speed, { firstBlock, block, stop, lang });
      return;
    }
    yield* this.vocode(f.feats, f.frames, { firstBlock, block, stop });
  }

  // Vocoder: one conditioning call + four 40-sample steps per 10 ms frame.
  async *vocode(feats, frames, { firstBlock, block, stop }) {
    const ort = this.ort;
    let state = Object.fromEntries(STATE.map(([n, w]) => [n, this.f32(new Float32Array(w), [1, w])]));
    let de = 0;
    const ctx = new Float32Array(60), per = new BigInt64Array(3);
    const period = (v) => BigInt(Math.min(255, Math.max(32, Math.round(256 / Math.pow(2, Math.min(10, Math.max(-10, v)) + 1.5)))));
    let size = Math.min(firstBlock, frames), out = new Float32Array(size * FRAME), filled = 0;
    for (let i = 0; i < frames; i++) {
      if (i % 10 === 0) {
        await tick();
        if (stop()) return;
      }
      const rows = [Math.max(i - 1, 0), i, Math.min(i + 1, frames - 1)];
      rows.forEach((r, j) => { ctx.set(feats.subarray(r * 20, r * 20 + 20), j * 20); per[j] = period(feats[r * 20 + 18]); });
      const cond = (await this.s.conditioning.run({
        features: this.f32(ctx.slice(), [1, 3, 20]), periods: new ort.Tensor("int64", per.slice(), [1, 3]),
      })).conditioning.data;
      for (let sub = 0; sub < 4; sub++) {
        const r = await this.s.vocoder_step.run({
          conditioning: this.f32(cond.slice(sub * 80, sub * 80 + 80), [1, 80]),
          period: new ort.Tensor("int64", BigInt64Array.from([per[1]]), [1]), ...state,
        });
        state = Object.fromEntries(STATE.map(([n]) => [n, r[n + "_out"]]));
        const x = r.excitation.data, base = filled * FRAME + sub * 40;
        for (let k = 0; k < 40; k++) { de = x[k] + 0.85 * de; out[base + k] = de; }
      }
      if (++filled === size) {
        yield out;
        size = Math.min(block, frames - i - 1);
        out = new Float32Array(Math.max(size, 0) * FRAME);
        filled = 0;
      }
    }
  }
}
