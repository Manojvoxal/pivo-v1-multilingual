// Streaming audio player: plays PCM blocks as they arrive from the worker, then lets the user
// pause, seek and replay the finished clip. Web Audio, no media element, no network.

export class StreamPlayer {
  constructor(sampleRate = 16000) {
    this.rate = sampleRate;
    this.ctx = null;
    this.reset();
  }

  /** Must be called from a click (browsers only allow audio after a user gesture). */
  unlock() {
    if (!this.ctx) {
      // iPhone: play through the silent switch like a media app (Safari 16.4+)
      try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch { /* not supported */ }
      const AC = window.AudioContext || window.webkitAudioContext;
      try { this.ctx = new AC({ sampleRate: this.rate, latencyHint: "playback" }); }
      catch { this.ctx = new AC({ latencyHint: "playback" }); }
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  reset() {
    this.stopSources();
    this.blocks = [];        // Float32Array PCM blocks in order
    this.total = 0;          // samples received
    this.complete = false;
    this.playing = false;
    this.ended = false;
    this.segments = [];      // { at: ctx time, offset: sample, length } for scheduled audio
    this.nextAt = 0;         // ctx time where the next block plays
    this.nextOffset = 0;     // sample offset of the next block to schedule
    this.endedOffset = 0;
  }

  stopSources() {
    for (const s of this.sources || []) { try { s.onended = null; s.stop(); s.disconnect(); } catch { /* already stopped */ } }
    this.sources = [];
  }

  /** Start a new clip; playback begins as soon as audio arrives. */
  begin() {
    this.reset();
    this.unlock();
    this.playing = true;
    this.ctx.resume();
    this.nextAt = this.ctx.currentTime + 0.12;   // small cushion against the first underrun
  }

  push(pcm) {
    this.blocks.push(pcm);
    this.total += pcm.length;
    if (this.segments.length || this.playing) {
      if (this.nextOffset === this.total - pcm.length) this.schedule(pcm, this.nextOffset);
    }
  }

  finish() {
    this.complete = true;
  }

  schedule(pcm, offset) {
    const ctx = this.ctx;
    const buf = ctx.createBuffer(1, pcm.length, this.rate);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const at = Math.max(this.nextAt, ctx.currentTime + 0.02);   // late block = short gap, never overlap
    src.start(at);
    this.sources.push(src);
    src.onended = () => { this.sources = this.sources.filter((s) => s !== src); };
    this.segments.push({ at, offset, length: pcm.length });
    if (this.segments.length > 400) this.segments.splice(0, this.segments.length - 400);
    this.nextAt = at + pcm.length / this.rate;
    this.nextOffset = offset + pcm.length;
  }

  /** All audio received so far, as one array. */
  audio() {
    const out = new Float32Array(this.total);
    let at = 0;
    for (const b of this.blocks) { out.set(b, at); at += b.length; }
    return out;
  }

  /** Current position in samples. */
  position() {
    if (this.ended) return this.endedOffset;
    if (!this.ctx || !this.segments.length) return this.nextOffset;
    const now = this.ctx.currentTime;
    let seg = this.segments[0];
    if (now < seg.at) return seg.offset;
    for (let i = this.segments.length - 1; i >= 0; i--) {
      if (this.segments[i].at <= now) { seg = this.segments[i]; break; }
    }
    return seg.offset + Math.min((now - seg.at) * this.rate, seg.length);
  }

  /** Called every animation frame by the UI; detects the end of playback. */
  update() {
    if (this.playing && this.complete && !this.ended && this.position() >= this.total - 1) {
      this.ended = true;
      this.playing = false;
      this.endedOffset = this.total;
      this.stopSources();
      this.segments = [];
      this.ctx.suspend();
    }
  }

  play() {
    this.unlock();
    if (this.ended || (this.complete && this.position() >= this.total - 1)) this.seek(0);
    this.playing = true;
    this.ctx.resume();
  }

  pause() {
    if (!this.ctx) return;
    this.playing = false;
    this.ctx.suspend();
  }

  /** Jump to a sample offset within the audio received so far. */
  seek(offset) {
    if (!this.ctx) return;
    offset = Math.max(0, Math.min(Math.floor(offset), this.total));
    this.stopSources();
    this.segments = [];
    this.ended = false;
    this.nextOffset = offset;
    this.nextAt = this.ctx.currentTime + 0.03;
    // one buffer for everything already received after the offset, then later blocks as they arrive
    if (offset < this.total) {
      const all = this.audio();
      this.schedule(all.subarray(offset), offset);
    } else {
      this.segments.push({ at: this.nextAt, offset, length: 0 });
    }
  }
}

/** Float32 PCM -> 16-bit mono WAV Blob (same scaling as pivo.audio.write_wav). */
export function wavBlob(pcm, sampleRate = 16000) {
  const data = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const text = (at, s) => { for (let i = 0; i < s.length; i++) data.setUint8(at + i, s.charCodeAt(i)); };
  text(0, "RIFF"); data.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, 1, true);
  data.setUint32(24, sampleRate, true); data.setUint32(28, sampleRate * 2, true);
  data.setUint16(32, 2, true); data.setUint16(34, 16, true);
  text(36, "data"); data.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const v = Math.max(-1, Math.min(1, pcm[i]));
    data.setInt16(44 + i * 2, Math.round(v * 32767), true);
  }
  return new Blob([data], { type: "audio/wav" });
}
