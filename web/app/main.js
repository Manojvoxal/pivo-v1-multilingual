// Voxal Studio page logic: text box, model loading status, generate / stop, player, download.
import { StreamPlayer, wavBlob } from "./player.js";

const $ = (id) => document.getElementById(id);
const MAX_CHARS = 1000;
const RATE = 16000;
// The two voices on the site. `voice` is the model voice id (see web/build.py SITE_VOICES).
const LANGS = {
  kn: {
    voice: "kn_m", name: "Kannada", avatar: "ಕ", native: "ಕನ್ನಡ", file: "kannada", htmlLang: "kn",
    placeholder: "ಇಲ್ಲಿ ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ… (Type or paste Kannada text here)",
    text: "ನಮಸ್ಕಾರ! ವೋಕ್ಸಲ್ ಸ್ಟುಡಿಯೋಗೆ ಸ್ವಾಗತ. ಈ ಧ್ವನಿ ಸಂಪೂರ್ಣವಾಗಿ ನಿಮ್ಮ ಕಂಪ್ಯೂಟರ್‌ನಲ್ಲೇ ತಯಾರಾಗುತ್ತದೆ.",
  },
  en: {
    voice: "en_m", name: "English", avatar: "En", native: "English", file: "english", htmlLang: "en",
    placeholder: "Type or paste English text here…",
    text: "Hello! Welcome to Voxal Studio. This voice is generated right here on your own computer.",
  },
};
const HAS_KANNADA = /[\u0C80-\u0CFF]/;

const store = {
  get(key) { try { return localStorage.getItem("voxal." + key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem("voxal." + key, value); } catch { /* private mode */ } },
};

// ------------------------------------------------------------------ elements
const textEl = $("text"), counterEl = $("counter"), genBtn = $("generate-btn"), genLabel = $("generate-label");
const speedEl = $("speed"), speedOut = $("speed-out");
const playerEl = $("player"), emptyEl = $("output-empty"), playBtn = $("play-btn"), wave = $("wave");
const timeNow = $("time-now"), timeTotal = $("time-total"), downloadBtn = $("download-btn");
const statsEl = $("stats"), readAs = $("read-as"), readAsText = $("read-as-text"), readAsNote = $("read-as-note"), errorEl = $("error");
const statusDot = $("status-dot"), statusTitle = $("status-title"), statusNote = $("status-note"), meter = $("load-meter"), deviceCard = $("device-card");

const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
$("generate-kbd").textContent = isMac ? "⌘ ↵" : "Ctrl ↵";

// ------------------------------------------------------------------ state
const player = new StreamPlayer(RATE);
let worker = null;
const state = {
  ready: false, loadFailed: false, busy: false, pending: false, id: 0,
  peaks: [],            // peak |x| per 10 ms frame of the current clip
  loudest: 0,
  estimate: 0,          // expected clip length in seconds while generating
  startedAt: 0,
  lang: "kn",
  clipLang: "kn",       // language of the clip in the player (for the file name)
};

// ------------------------------------------------------------------ text box
function onTextChange() {
  const n = textEl.value.length;
  counterEl.textContent = `${n.toLocaleString("en-IN")} / ${MAX_CHARS.toLocaleString("en-IN")} characters`;
  counterEl.classList.toggle("near", n > MAX_CHARS * 0.9);
  store.set("text", textEl.value);
  updateLangTip();
}

// ------------------------------------------------------------------ language
const langRadios = [...document.querySelectorAll('input[name="lang"]')];
const voiceSelect = $("voice-select");

function setLang(lang) {
  if (!LANGS[lang]) lang = "kn";
  const prev = LANGS[state.lang], next = LANGS[lang];
  state.lang = lang;
  store.set("lang", lang);
  for (const r of langRadios) r.checked = r.value === lang;
  voiceSelect.value = lang;
  for (const el of [$("chip-avatar"), $("voice-avatar")]) {
    el.textContent = next.avatar;
    el.classList.toggle("is-latin", lang === "en");
  }
  $("voice-name").textContent = next.name;
  $("voice-meta").textContent = `Narration voice · ${next.native}`;
  textEl.placeholder = next.placeholder;
  textEl.lang = next.htmlLang;
  textEl.classList.toggle("kn", lang === "kn");
  // swap the sample sentence, but never the user's own text
  if (prev !== next && textEl.value.trim() === prev.text) { textEl.value = next.text; onTextChange(); }
  if (lang === "en") worker?.postMessage({ type: "prefetch" });
  updateLangTip();
}

// A gentle hint when the text and the chosen voice do not match.
function updateLangTip() {
  const text = textEl.value, tip = $("lang-tip");
  let other = null;
  if (state.lang === "en" && HAS_KANNADA.test(text)) other = "kn";
  else if (state.lang === "kn" && !HAS_KANNADA.test(text) && (text.match(/[A-Za-z]/g) || []).length >= 8) other = "en";
  tip.hidden = !other;
  if (other) {
    $("lang-tip-text").textContent = other === "kn" ? "This text is in Kannada." : "This text looks like English.";
    $("lang-tip-btn").textContent = `Use the ${LANGS[other].name} voice`;
    $("lang-tip-btn").dataset.lang = other;
  }
}

for (const r of langRadios) r.addEventListener("change", () => r.checked && setLang(r.value));
voiceSelect.addEventListener("change", () => setLang(voiceSelect.value));
$("lang-tip-btn").addEventListener("click", (e) => setLang(e.currentTarget.dataset.lang));

state.lang = LANGS[store.get("lang")] ? store.get("lang") : "kn";
textEl.value = store.get("text") ?? LANGS[state.lang].text;
setLang(state.lang);
onTextChange();
textEl.addEventListener("input", onTextChange);
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); onGenerate(); }
});
$("clear-btn").addEventListener("click", () => { textEl.value = ""; onTextChange(); textEl.focus(); });

// ------------------------------------------------------------------ speed
function onSpeed() {
  const v = Number(speedEl.value);
  speedOut.textContent = `${v.toFixed(v * 10 % 1 ? 2 : 1)}×`;
  speedEl.style.setProperty("--fill", `${((v - 0.5) / 1.5) * 100}%`);
  store.set("speed", String(v));
}
const savedSpeed = Number(store.get("speed"));
if (savedSpeed >= 0.5 && savedSpeed <= 2) speedEl.value = String(savedSpeed);
onSpeed();
speedEl.addEventListener("input", onSpeed);
$("speed-reset").addEventListener("click", () => { speedEl.value = "1"; onSpeed(); });

// ------------------------------------------------------------------ dialogs
for (const btn of document.querySelectorAll("[data-open]")) {
  btn.addEventListener("click", () => $(btn.dataset.open).showModal());
}
for (const dialog of document.querySelectorAll("dialog")) {
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });   // backdrop click
}

// ------------------------------------------------------------------ worker

function setStatus(kind, title, note) {
  statusDot.className = "status-dot" + (kind === "ready" ? " ready" : kind === "failed" ? " failed" : "");
  deviceCard.classList.toggle("is-ready", kind === "ready");
  statusTitle.textContent = title;
  statusNote.textContent = "";
  if (typeof note === "string") statusNote.textContent = note;
  else if (note) statusNote.append(...note);
}

function browserProblem() {
  if (typeof WebAssembly !== "object") return "This browser cannot run WebAssembly.";
  try { new RegExp("(?<!a)b", "u"); } catch { return "This browser is too old for the Kannada text reader."; }
  if (!("AudioContext" in window || "webkitAudioContext" in window)) return "This browser cannot play generated audio.";
  return null;
}

function startWorker() {
  const problem = browserProblem();
  if (problem) return loadFailed(`${problem} Please use a recent Chrome, Edge, Firefox or Safari.`, false);
  state.loadFailed = false;
  setStatus("loading", "Loading model…", "One-time download of about 9 MB. After that it opens instantly.");
  meter.style.width = "2%";
  try {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  } catch {
    return loadFailed("This browser does not support module workers. Please use a recent Chrome, Edge, Firefox or Safari.", false);
  }
  worker.onmessage = (e) => onWorker(e.data);
  worker.onerror = (e) => { e.preventDefault?.(); loadFailed(state.ready ? "The voice engine stopped unexpectedly." : "The voice engine could not start in this browser."); };
  worker.postMessage({ type: "load" });
}

function loadFailed(message, canRetry = true) {
  state.ready = false;
  state.loadFailed = true;
  state.pending = false;
  if (state.busy) endGeneration();
  worker?.terminate();
  worker = null;
  const parts = [document.createTextNode(message + " ")];
  if (canRetry) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "link-btn";
    retry.textContent = "Try again";
    retry.addEventListener("click", startWorker);
    parts.push(retry);
  }
  setStatus("failed", "Model not loaded", parts);
  updateButton();
}

function onWorker(msg) {
  switch (msg.type) {
    case "progress":
      if (msg.stage === "download") {
        meter.style.width = `${Math.max(2, msg.fraction * 100).toFixed(1)}%`;
        statusTitle.textContent = `Loading model… ${Math.round(msg.fraction * 100)}%`;
      } else if (msg.stage === "prepare") {
        meter.style.width = "100%";
        statusTitle.textContent = "Preparing the voice…";
      }
      updateButton();
      break;
    case "ready":
      state.ready = true;
      setStatus("ready", "Ready on your device", `Model loaded in ${msg.seconds.toFixed(1)} s. Everything runs locally.`);
      updateButton();
      if (state.lang === "en") worker.postMessage({ type: "prefetch" });
      if (state.pending) { state.pending = false; generate(); }
      break;
    case "error":
      if (msg.stage === "load") loadFailed(friendlyLoadError(msg.message));
      else if (msg.id === state.id) { showError(`Could not generate audio: ${msg.message}`); endGeneration(); }
      break;
    default:
      if (msg.id !== state.id) return;            // a message from an older, cancelled request
      onGeneration(msg);
  }
}

function friendlyLoadError(message) {
  if (/fetch|network|download/i.test(message)) return "Could not download the model. Check your internet connection.";
  return `Could not start the model (${message}).`;
}

// ------------------------------------------------------------------ generate
function updateButton() {
  genBtn.classList.toggle("is-busy", state.busy || state.pending);
  if (state.busy) genLabel.textContent = "Stop";
  else if (state.pending) genLabel.textContent = "Starting soon…";
  else genLabel.textContent = "Generate";
  genBtn.disabled = state.loadFailed && !state.busy;
  genBtn.setAttribute("aria-label", state.busy ? "Stop generating" : "Generate speech");
}

function onGenerate() {
  if (state.busy) return stop();
  if (state.pending) { state.pending = false; updateButton(); return; }
  hideError();
  if (!textEl.value.trim()) {
    showError(`Type some ${LANGS[state.lang].name} text first.`);
    textEl.focus();
    return;
  }
  player.unlock();                                   // inside the click: allowed to start audio
  if (!state.ready) {
    if (state.loadFailed) startWorker();
    state.pending = true;
    updateButton();
    statsEl.textContent = "Generation will start as soon as the model has loaded.";
    return;
  }
  generate();
}
genBtn.addEventListener("click", onGenerate);

function generate() {
  state.id += 1;
  state.busy = true;
  state.peaks = [];
  state.loudest = 0;
  state.estimate = 0;
  state.startedAt = performance.now();
  hideError();
  emptyEl.hidden = true;
  playerEl.hidden = false;
  readAs.hidden = true;
  downloadBtn.disabled = true;
  statsEl.textContent = "Generating on your device…";
  player.begin();
  updateButton();
  state.clipLang = state.lang;
  worker.postMessage({ type: "synth", id: state.id, text: textEl.value.slice(0, MAX_CHARS),
    speed: Number(speedEl.value), voice: LANGS[state.lang].voice });
  loop();
}

function stop() {
  worker?.postMessage({ type: "cancel", id: state.id });
  player.finish();
  endGeneration();
  statsEl.textContent = player.total ? "Stopped. You can play or download what was made so far." : "Stopped.";
  downloadBtn.disabled = player.total === 0;
}

function endGeneration() {
  state.busy = false;
  updateButton();
}

function onGeneration(msg) {
  switch (msg.type) {
    case "status":
      statsEl.textContent = msg.message;
      break;
    case "text": {
      state.estimate = msg.estimate || 0;
      readAsText.textContent = msg.normalized || "(nothing to read)";
      const notes = [];
      if (msg.skipped.length) notes.push(`Skipped (cannot be read aloud): ${msg.skipped.join(" ")}`);
      if (msg.unknown.length) notes.push(`Not in the dictionary, read by Indian spelling rules: ${msg.unknown.join(", ")}`);
      readAsNote.textContent = notes.join("\n");
      readAsNote.hidden = !notes.length;
      readAs.hidden = false;
      if (msg.skipped.length) readAs.open = true;
      break;
    }
    case "audio":
      player.push(msg.pcm);
      addPeaks(msg.pcm);
      break;
    case "done":
      player.finish();
      endGeneration();
      if (msg.empty) {
        statsEl.textContent = "";
        showError("There is nothing to read aloud in this text.");
        playerEl.hidden = true;
        emptyEl.hidden = false;
        return;
      }
      downloadBtn.disabled = false;
      statsEl.innerHTML = "";
      statsEl.append(...statsLine(msg));
      break;
    case "cancelled":
      break;
  }
}

function statsLine({ audioSeconds, seconds }) {
  const strong = (t) => { const s = document.createElement("strong"); s.textContent = t; return s; };
  const speedup = audioSeconds / Math.max(seconds, 0.001);
  return [strong(`${audioSeconds.toFixed(1)} s`), document.createTextNode(" of audio, made in "),
    strong(`${seconds.toFixed(1)} s`), document.createTextNode(` on your device (${speedup >= 10 ? Math.round(speedup) : speedup.toFixed(1)}× faster than real time).`)];
}

function showError(text) { errorEl.textContent = text; errorEl.hidden = false; }
function hideError() { errorEl.hidden = true; }

// ------------------------------------------------------------------ player controls
playBtn.addEventListener("click", () => {
  if (player.playing) player.pause();
  else player.play();
  loop();
});

downloadBtn.addEventListener("click", () => {
  if (!player.total) return;
  const url = URL.createObjectURL(wavBlob(player.audio(), RATE));
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
  a.href = url;
  a.download = `voxal-${LANGS[state.clipLang].file}-${stamp}.wav`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
});

const displaySeconds = () => Math.max(player.total / RATE, state.busy ? state.estimate : 0, 0.01);

function seekToFraction(f) {
  if (!player.total) return;
  const target = Math.min(Math.max(f, 0), 1) * displaySeconds() * RATE;
  player.seek(Math.min(target, player.total));
  loop();
}
wave.addEventListener("pointerdown", (e) => {
  const rect = wave.getBoundingClientRect();
  seekToFraction((e.clientX - rect.left) / rect.width);
  wave.setPointerCapture(e.pointerId);
  const move = (ev) => seekToFraction((ev.clientX - rect.left) / rect.width);
  wave.addEventListener("pointermove", move);
  wave.addEventListener("pointerup", () => wave.removeEventListener("pointermove", move), { once: true });
});
wave.addEventListener("keydown", (e) => {
  const step = 2 * RATE;
  const pos = player.position();
  if (e.key === "ArrowRight") player.seek(Math.min(pos + step, player.total));
  else if (e.key === "ArrowLeft") player.seek(Math.max(pos - step, 0));
  else if (e.key === "Home") player.seek(0);
  else if (e.key === "End") player.seek(player.total);
  else return;
  e.preventDefault();
  loop();
});
document.addEventListener("keydown", (e) => {
  if (e.key !== " " || playerEl.hidden || !player.total) return;
  if (e.target.closest("textarea, input, button, select, summary, dialog, a")) return;
  e.preventDefault();
  playBtn.click();
});

// ------------------------------------------------------------------ waveform
function addPeaks(pcm) {
  for (let i = 0; i < pcm.length; i += 160) {
    let m = 0;
    const end = Math.min(i + 160, pcm.length);
    for (let k = i; k < end; k++) { const v = Math.abs(pcm[k]); if (v > m) m = v; }
    state.peaks.push(m);
    if (m > state.loudest) state.loudest = m;
  }
}

const fmt = (s) => { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = wave.clientWidth, h = wave.clientHeight;
  if (!w) return;
  if (wave.width !== Math.round(w * dpr) || wave.height !== Math.round(h * dpr)) {
    wave.width = Math.round(w * dpr);
    wave.height = Math.round(h * dpr);
  }
  const g = wave.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const barW = 3, gap = 2, bars = Math.max(1, Math.floor((w + gap) / (barW + gap)));
  const framesShown = displaySeconds() * 100;           // 100 peaks per second
  const perBar = framesShown / bars;
  const posFrames = player.position() / 160;
  const loudest = Math.max(0.05, state.loudest);
  for (let b = 0; b < bars; b++) {
    const from = Math.floor(b * perBar), to = Math.max(from + 1, Math.floor((b + 1) * perBar));
    let m = 0, any = false;
    for (let k = from; k < to && k < state.peaks.length; k++) { any = true; if (state.peaks[k] > m) m = state.peaks[k]; }
    const x = b * (barW + gap);
    const bh = any ? Math.max(3, Math.sqrt(m / loudest) * (h - 6)) : 3;
    g.fillStyle = !any ? "#efedea" : from < posFrames ? "#121212" : "#d4d0ca";
    roundRect(g, x, (h - bh) / 2, barW, bh, 1.5);
  }
}
function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h);
  g.fill();
}

let looping = false;
function loop() {
  if (looping) return;
  looping = true;
  const frame = () => {
    player.update();
    const pos = player.position() / RATE, total = player.total / RATE;
    timeNow.textContent = fmt(pos);
    timeTotal.textContent = fmt(state.busy ? Math.max(total, state.estimate) : total);
    playBtn.classList.toggle("is-playing", player.playing);
    playBtn.setAttribute("aria-label", player.playing ? "Pause" : "Play");
    wave.setAttribute("aria-valuemax", total.toFixed(1));
    wave.setAttribute("aria-valuenow", pos.toFixed(1));
    wave.setAttribute("aria-valuetext", `${fmt(pos)} of ${fmt(total)}`);
    draw();
    if (player.playing || state.busy) requestAnimationFrame(frame);
    else looping = false;
  };
  requestAnimationFrame(frame);
}
new ResizeObserver(() => { if (!playerEl.hidden) draw(); }).observe(wave);

startWorker();
