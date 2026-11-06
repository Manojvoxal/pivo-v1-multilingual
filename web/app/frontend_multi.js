// PiVo-Multi text frontend for the browser (metadata frontend "pivo-multi-v1").
//
// A line-by-line port of training/data_pipeline_multi/pivo_multi/{symbols,scripts,langs,brahmic,
// frontend}.py: the same text must give the same token ids as the Python frontend that labelled
// the training data. web/tests/parity_multi.mjs checks that on the validation manifest.
//
// On top of the training frontend (which rejects numbers), the web app reads numbers, dates,
// times, money, units and symbols aloud: Kannada with PiVo's Kannada reader (frontend.js
// normalizeKannadaText), English the Indian way (normalizeIndianEnglish), and Hindi, Marathi,
// Gujarati, Punjabi, Bengali, Odia, Assamese, Tamil, Telugu and Malayalam in their own words
// (numbers_multi.js, a port of pivo_multi/numnorm.py). User text never throws in lenient mode:
// what cannot be read is skipped and listed.

import {
  lookup, romanizedArpabet, normalizeEnglish, normalizeIndianEnglish, normalizeKannadaText,
  indianEnglishCardinal, cardinal, SPECIAL,
} from "./frontend.js";
import { normalizeNumbers, NUMBER_LANGS } from "./numbers_multi.js";

export const FRONTEND_NAME = "pivo-multi-v1";

// ------------------------------------------------------------------ symbols (symbols.py)
const CONSONANTS_EN = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(" ");
const VOWELS_EN = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(" ");
const ARPABET = [...CONSONANTS_EN, ...VOWELS_EN.flatMap((v) => [v + "0", v + "1", v + "2"])];
const IN_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au", "ae", "aw", "ix"];
const IN_NASAL_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au", "ae", "aw"].map((v) => v + "~");
const IN_CONSONANTS = [
  "k", "kh", "g", "gh", "ng", "c", "ch", "j", "jh", "ny", "tt", "tth", "dd", "ddh", "nn",
  "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "ll",
  "sh", "ssh", "s", "h", "f", "z", "q", "x", "gx", "rr", "rrh", "zh", "rx", "tx", "nx",
  "bq", "dq", "jq", "gq",
];
export const SYMBOLS = [...SPECIAL, ...ARPABET, ...[...IN_VOWELS, ...IN_NASAL_VOWELS, ...IN_CONSONANTS].map((p) => "in_" + p)];
export const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]));
const IN_VOWEL_SET = new Set(IN_VOWELS);
const IN_STOPS_BY_PLACE = {};
for (const [nasal, group] of [["ng", "k kh g gh q x gx"], ["ny", "c ch j jh"], ["nn", "tt tth dd ddh"],
  ["n", "t th d dh"], ["m", "p ph b bh f"]]) {
  for (const p of group.split(" ")) IN_STOPS_BY_PLACE[p] = nasal;
}
const nasalize = (p) => (IN_VOWEL_SET.has(p) && p !== "ix" ? p + "~" : p);

// ------------------------------------------------------------------ scripts (scripts.py)
const BRAHMIC_BASE = {
  devanagari: 0x0900, bengali: 0x0980, gurmukhi: 0x0A00, gujarati: 0x0A80, odia: 0x0B00,
  tamil: 0x0B80, telugu: 0x0C00, kannada: 0x0C80, malayalam: 0x0D00,
};
const OTHER_RANGES = {
  olchiki: [[0x1C50, 0x1C7F]],
  meetei: [[0xABC0, 0xABFF], [0xAAE0, 0xAAFF]],
  arabic: [[0x0600, 0x06FF], [0x0750, 0x077F], [0x08A0, 0x08FF], [0xFB50, 0xFDFF], [0xFE70, 0xFEFF]],
  latin: [[0x41, 0x5A], [0x61, 0x7A]],
};
const ZWNJ = "‌", ZWJ = "‍";

export function scriptOfChar(ch) {
  const cp = ch.codePointAt(0);
  for (const [name, base] of Object.entries(BRAHMIC_BASE)) if (cp >= base && cp < base + 0x80) return name;
  for (const [name, ranges] of Object.entries(OTHER_RANGES)) {
    for (const [lo, hi] of ranges) if (cp >= lo && cp <= hi) return name;
  }
  if ((cp >= 0x1CD0 && cp <= 0x1CFF) || (cp >= 0xA8E0 && cp <= 0xA8FF)) return "devanagari";
  return null;
}

function scriptOfWord(word) {
  const found = new Set();
  for (const ch of word) {
    if (ch === ZWJ || ch === ZWNJ) continue;
    const s = scriptOfChar(ch);
    if (s === null) return null;
    found.add(s);
  }
  return found.size === 1 ? [...found][0] : null;
}

// ------------------------------------------------------------------ languages (langs.py)
//            code   script        inherent schwa     anusvara     merge_len vocalic_r
const LANG_ROWS = [
  ["en", "latin", null, null, null, null, null],
  ["kn", "kannada", "a", "none", "dravidian", null, ["r", "u"]],
  ["hi", "devanagari", "a", "medial", "indo", null, ["r", "i"]],
  ["te", "telugu", "a", "none", "dravidian", null, ["r", "u"]],
  ["ta", "tamil", "a", "none", "dravidian", null, ["r", "u"]],
  ["bn", "bengali", "aw", "medial", "velar", "all", ["r", "i"]],
  ["mr", "devanagari", "a", "medial", "indo", null, ["r", "u"]],
  ["gu", "gujarati", "a", "medial", "indo", null, ["r", "u"]],
  ["pa", "gurmukhi", "a", "medial", "indo", null, ["r", "i"]],
  ["ml", "malayalam", "a", "none", "dravidian", null, ["r", "u"]],
  ["or", "odia", "aw", "none", "velar", "all", ["r", "u"]],
  ["as", "bengali", "aw", "final", "velar", "all", ["r", "i"]],
  ["ne", "devanagari", "a", "final", "indo", "iu", ["r", "i"]],
  ["kok", "devanagari", "a", "final", "indo", null, ["r", "u"]],
  ["mai", "devanagari", "a", "medial", "indo", null, ["r", "i"]],
  ["doi", "devanagari", "a", "medial", "indo", null, ["r", "i"]],
  ["sa", "devanagari", "a", "none", "dravidian", null, ["r", "i"]],
  ["brx", "devanagari", "a", "final", "indo", null, ["r", "i"]],
  ["mni", "meetei", "a", "none", "indo", "all", ["r", "i"]],
  ["sat", "olchiki", null, null, null, null, null],
  ["ur", "arabic", null, null, null, null, null],
  ["ks", "arabic", null, null, null, null, null],
  ["sd", "arabic", null, null, null, null, null],
];
export const LANG = Object.fromEntries(LANG_ROWS.map(([code, script, inherent, schwa, anusvara, mergeLen, vocalicR]) =>
  [code, { code, script, inherent, schwa, anusvara, mergeLen, vocalicR }]));
export const LANG_TO_ID = Object.fromEntries(LANG_ROWS.map((r, i) => [r[0], i]));
export const SCRIPT_DEFAULT_LANG = {
  devanagari: "hi", bengali: "bn", gurmukhi: "pa", gujarati: "gu", odia: "or", tamil: "ta",
  telugu: "te", kannada: "kn", malayalam: "ml", olchiki: "sat", meetei: "mni",
};
const LANG_SCRIPTS = { mni: ["meetei", "bengali"], sat: ["olchiki", "devanagari"], ks: ["arabic", "devanagari"], sd: ["arabic", "devanagari"] };
const langScripts = (code) => LANG_SCRIPTS[code] || [LANG[code].script];

// ------------------------------------------------------------------ Brahmic G2P (brahmic.py)
const IND_VOWEL = { 0x05: "@IND", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "VR", 0x0C: "VL",
  0x0D: "ae", 0x0E: "e", 0x0F: "ee", 0x10: "ai", 0x11: "aw", 0x12: "o", 0x13: "oo", 0x14: "au", 0x60: "VRR", 0x61: "VLL" };
const MATRA = { 0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "VR", 0x44: "VRR", 0x45: "ae",
  0x46: "e", 0x47: "ee", 0x48: "ai", 0x49: "aw", 0x4A: "o", 0x4B: "oo", 0x4C: "au", 0x62: "VL", 0x63: "VLL" };
const CONS = { 0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "c", 0x1B: "ch", 0x1C: "j", 0x1D: "jh",
  0x1E: "ny", 0x1F: "tt", 0x20: "tth", 0x21: "dd", 0x22: "ddh", 0x23: "nn", 0x24: "t", 0x25: "th", 0x26: "d",
  0x27: "dh", 0x28: "n", 0x29: "nx", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
  0x31: "rx", 0x32: "l", 0x33: "ll", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "ssh", 0x38: "s", 0x39: "h" };
const CANDRABINDU = 0x01, ANUSVARA = 0x02, VISARGA = 0x03, NUKTA = 0x3C, AVAGRAHA = 0x3D, VIRAMA = 0x4D, OM = 0x50;
const DANDA_CPS = new Set([0x0964, 0x0965]);
const NUKTA_MAP = { k: "q", kh: "x", g: "gx", j: "z", jh: "z", dd: "rr", ddh: "rrh", ph: "f" };
const EXTRA = {
  devanagari: { 0x58: ["C", "q"], 0x59: ["C", "x"], 0x5A: ["C", "gx"], 0x5B: ["C", "z"], 0x5C: ["C", "rr"],
    0x5D: ["C", "rrh"], 0x5E: ["C", "f"], 0x5F: ["C", "y"], 0x3A: ["M", "o"], 0x3B: ["M", "oo"], 0x4E: ["IGN"],
    0x4F: ["M", "aw"], 0x55: ["IGN"], 0x56: ["M", "u"], 0x57: ["M", "uu"], 0x70: ["IGN"], 0x71: ["IGN"],
    0x72: ["V", "ae"], 0x73: ["V", "o"], 0x74: ["V", "oo"], 0x75: ["V", "aw"], 0x76: ["V", "u"], 0x77: ["V", "uu"],
    0x79: ["C", "z"], 0x7A: ["C", "y"], 0x7B: ["C", "gq"], 0x7C: ["C", "jq"], 0x7D: ["IGN"], 0x7E: ["C", "dq"],
    0x7F: ["C", "bq"], 0x00: ["NAS", "candrabindu"] },
  bengali: { 0x5C: ["C", "rr"], 0x5D: ["C", "rrh"], 0x5F: ["C", "y"], 0x4E: ["C0", "t"], 0x70: ["C", "r"],
    0x71: ["C", "v"], 0x57: ["IGN"], 0x72: ["IGN"], 0x73: ["IGN"], 0x7A: ["IGN"], 0x7C: ["IGN"], 0x7D: ["IGN"], 0x7E: ["IGN"] },
  gurmukhi: { 0x59: ["C", "x"], 0x5A: ["C", "gx"], 0x5B: ["C", "z"], 0x5C: ["C", "rr"], 0x5E: ["C", "f"],
    0x70: ["NAS", "anusvara"], 0x71: ["GEM"], 0x72: ["BEARER", "i"], 0x73: ["BEARER", "u"], 0x74: ["IGN"],
    0x75: ["C", "y"], 0x51: ["IGN"] },
  gujarati: { 0x7A: ["IGN"], 0x7B: ["IGN"], 0x7C: ["IGN"], 0x7D: ["IGN"], 0x7E: ["IGN"], 0x7F: ["IGN"] },
  odia: { 0x5C: ["C", "rr"], 0x5D: ["C", "rrh"], 0x5F: ["C", "y"], 0x71: ["C", "v"], 0x56: ["IGN"], 0x57: ["IGN"], 0x70: ["IGN"] },
  tamil: { 0x57: ["IGN"] },
  telugu: { 0x58: ["C", "c"], 0x59: ["C", "j"], 0x5A: ["C", "r"], 0x5D: ["C0", "n"], 0x00: ["IGN"], 0x01: ["IGN"],
    0x04: ["IGN"], 0x55: ["IGN"], 0x56: ["IGN"] },
  kannada: { 0x5D: ["C0", "n"], 0x5E: ["C", "ll"], 0x55: ["IGN"], 0x56: ["IGN"], 0x71: ["IGN"], 0x72: ["IGN"],
    0x00: ["IGN"], 0x04: ["IGN"] },
  malayalam: { 0x54: ["C0", "m"], 0x55: ["C0", "y"], 0x56: ["C0", "zh"], 0x7A: ["C0", "nn"], 0x7B: ["C0", "nx"],
    0x7C: ["C0", "r"], 0x7D: ["C0", "l"], 0x7E: ["C0", "ll"], 0x7F: ["C0", "k"], 0x4E: ["C0", "r"], 0x57: ["M", "au"],
    0x00: ["IGN"], 0x3B: ["IGN"], 0x3C: ["IGN"], 0x4F: ["IGN"] },
};
const IGNORE_OFFSETS = new Set([AVAGRAHA, 0x51, 0x52, 0x53, 0x54]);
const isDigitOffset = (o) => o >= 0x66 && o < 0x70;
const ASPIRATE_TO_PLAIN = { kh: "k", gh: "g", ch: "c", jh: "j", tth: "tt", ddh: "dd", th: "t", dh: "d", ph: "p", bh: "b", rrh: "rr" };
const NASALS = new Set(["ng", "ny", "nn", "n", "m", "nx"]);
const VOICELESS_OBSTRUENTS = new Set(["k", "c", "tt", "t", "p", "s", "sh", "ssh", "tx", "kh", "ch", "tth", "th", "ph", "f", "x", "q", "h"]);
const SIBILANT_CLUSTER_NEXT = new Set(["t", "th", "tt", "tth", "n", "nn", "l", "r", "k", "kh", "p", "ph", "m"]);
const OVERRIDES = {
  hi: { ssh: "sh" }, mai: { ssh: "sh" }, doi: { ssh: "sh" }, gu: { ssh: "sh" }, pa: { ssh: "sh" }, kok: { ssh: "sh" },
  brx: { ssh: "sh" }, ne: { sh: "s", ssh: "s" }, or: { sh: "s", ssh: "s" }, bn: { ssh: "sh", s: "sh", v: "b" },
  as: { c: "s", ch: "s", j: "z", jh: "z", tt: "t", tth: "th", dd: "d", ddh: "dh", nn: "n", sh: "x", ssh: "x", s: "x", rr: "r", rrh: "r" },
  mni: { ssh: "s", sh: "s", v: "b" },
};
const JNY = { hi: ["g", "y"], mai: ["g", "y"], doi: ["g", "y"], ne: ["g", "y"], kok: ["g", "y"], brx: ["g", "y"],
  pa: ["g", "y"], or: ["g", "y"], mr: ["d", "ny"], gu: ["g", "n"], bn: ["g", "g"], as: ["g", "g"], mni: ["g", "y"] };
const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
const item = (kind, phone = null, off = null) => ({ kind, phone, off, deleted: false });

function vowelPhones(v, cfg) {
  const rVowel = cfg.vocalicR[1];
  const longV = { i: "ii", u: "uu" }[rVowel];
  if (v === "VR") return ["r", rVowel];
  if (v === "VRR") return ["r", longV];
  if (v === "VL") return ["l", rVowel];
  if (v === "VLL") return ["l", longV];
  if (v === "@IND") return [cfg.inherent];
  return [v];
}

function parse(word, script, lang) {
  const base = BRAHMIC_BASE[script];
  const extra = EXTRA[script] || {};
  const items = [];
  const chars = Array.from(word);
  const n = chars.length;
  const off = (j) => chars[j].codePointAt(0) - base;
  const inBlock = (j) => { const cp = chars[j].codePointAt(0); return cp >= base && cp < base + 0x80; };

  const attachVowel = (j) => {
    while (j < n && chars[j] === ZWNJ) j++;
    if (j < n && off(j) === VIRAMA) {
      j++;
      if (j < n && chars[j] === ZWJ) return j + 1;
      if (script === "malayalam") {
        let k = j;
        while (k < n && (chars[k] === ZWNJ || chars[k] === ZWJ)) k++;
        if (k >= n) items.push(item("HALEND"));
      }
      return j;
    }
    if (j < n && inBlock(j)) {
      const o = off(j);
      if (has(MATRA, o)) { items.push(item("V", MATRA[o], o)); return j + 1; }
      const ex = extra[o];
      if (ex && ex[0] === "M") { items.push(item("V", ex[1], o)); return j + 1; }
    }
    items.push(item("@"));
    return j;
  };

  let i = 0;
  while (i < n) {
    const ch = chars[i];
    if (ch === ZWJ || ch === ZWNJ) { i++; continue; }
    const cp = ch.codePointAt(0);
    if (DANDA_CPS.has(cp)) { i++; continue; }
    if (!(cp >= base && cp < base + 0x80)) return null;
    let o = cp - base;
    if (isDigitOffset(o)) return null;
    const ex = has(extra, o) ? extra[o] : undefined;
    if (ex !== undefined) {
      const kind = ex[0];
      if (kind === "IGN") i++;
      else if (kind === "C") { items.push(item("C", ex[1], o)); i = attachVowel(i + 1); }
      else if (kind === "C0") { items.push(item("C", ex[1], o)); i++; }
      else if (kind === "V") { items.push(item("V", ex[1], o)); i++; }
      else if (kind === "M") i++;
      else if (kind === "NAS") { items.push(item("NAS", ex[1], o)); i++; }
      else if (kind === "GEM") { items.push(item("GEM", null, o)); i++; }
      else if (kind === "BEARER") {
        let j = i + 1;
        if (j < n && inBlock(j) && has(MATRA, off(j))) { items.push(item("V", MATRA[off(j)], o)); j++; }
        else items.push(item("V", ex[1], o));
        i = j;
      }
      continue;
    }
    if (has(CONS, o)) {
      let phone = CONS[o];
      if (o === 0x29) phone = lang === "ta" || lang === "ml" ? "nx" : "n";
      else if (o === 0x31) phone = lang === "ta" || lang === "ml" ? "rx" : "r";
      else if (o === 0x34) phone = lang === "ta" || lang === "ml" ? "zh" : "ll";
      let j = i + 1;
      if (j < n && off(j) === NUKTA) {
        phone = NUKTA_MAP[phone] ?? phone;
        if (o === 0x2F) o = 0x5F;          // য় / य़ (decomposed by NFC) = /y/, not য
        j++;
      }
      items.push(item("C", phone, o));
      i = attachVowel(j);
      continue;
    }
    if (has(IND_VOWEL, o)) { items.push(item("V", IND_VOWEL[o], o)); i++; continue; }
    if (o === ANUSVARA || o === CANDRABINDU) {
      items.push(item("NAS", o === ANUSVARA ? "anusvara" : "candrabindu", o));
      i++;
      continue;
    }
    if (o === VISARGA) { items.push(item(script === "tamil" ? "AYTHAM" : "VIS", null, o)); i++; continue; }
    if (o === OM) { items.push(item("V", "oo", o)); items.push(item("C", "m", o)); i++; continue; }
    if (IGNORE_OFFSETS.has(o) || o === NUKTA || o === VIRAMA || has(MATRA, o)) { i++; continue; }
    return null;
  }
  return items;
}

const slotsOf = (items) => items.flatMap((it, k) => (it.kind === "V" || it.kind === "@" ? [k] : []));

function onsetLen(items, slots, s) {
  const lo = s > 0 ? slots[s - 1] + 1 : 0;
  return items.slice(lo, slots[s]).filter((it) => ["C", "NAS", "GEM", "VIS"].includes(it.kind)).length;
}
function onsetPhones(items, slots, s) {
  const lo = s > 0 ? slots[s - 1] + 1 : 0;
  return items.slice(lo, slots[s]).filter((it) => it.kind === "C").map((it) => it.phone);
}

function deleteSchwas(items, mode, keepAfterCluster = false) {
  if (mode === null || mode === "none") return;
  const slots = slotsOf(items);
  if (slots.length < 2) return;
  const last = slots[slots.length - 1];
  const it = items[last];
  if (it.kind === "@" && last === items.length - 1) {
    const onset = onsetPhones(items, slots, slots.length - 1);
    const prev = items[slots[slots.length - 2]];
    let keep = false;
    if (onset.length === 1 && onset[0] === "y" && prev.kind === "V" && (prev.phone === "i" || prev.phone === "ii")) keep = true;
    else if (onset.length >= 2 && (["r", "y", "v"].includes(onset[onset.length - 1]) || keepAfterCluster)) keep = true;
    else if (!onset.length) keep = true;
    if (!keep) it.deleted = true;
  }
  if (mode !== "medial") return;
  for (let s = slots.length - 2; s > 0; s--) {
    const cur = items[slots[s]];
    if (cur.kind !== "@" || cur.deleted) continue;
    if (items[slots[s + 1]].deleted) continue;
    if (onsetLen(items, slots, s) === 1 && onsetLen(items, slots, s + 1) === 1) {
      if (items[slots[s] + 1].kind === "C") cur.deleted = true;
    }
  }
}

function clusterRules(items, lang) {
  const out = [];
  for (let it of items) {
    const prev = out.length ? out[out.length - 1] : null;
    const afterCons = it.kind === "C" && prev !== null && prev.kind === "C";
    if (afterCons) {
      const wordInitial = out.every((x) => x.kind === "C");
      const phala = (it.off === 0x2F && it.phone === "y" && prev.phone !== "r")
        || (it.off === 0x2C && it.phone === "b" && prev.phone !== "r" && prev.phone !== "m");
      if ((lang === "bn" || lang === "as") && phala) {
        if (!wordInitial) out.push(item("C", ASPIRATE_TO_PLAIN[prev.phone] ?? prev.phone, it.off));
        continue;
      }
      if (prev.phone === "j" && prev.off === 0x1C && it.phone === "ny" && has(JNY, lang)) {
        out.pop();
        for (const p of JNY[lang]) out.push(item("C", p, 0x1C));
        continue;
      }
      if (prev.phone === "k" && it.off === 0x37 && (lang === "bn" || lang === "as")) {
        out.pop();
        if (!wordInitial) out.push(item("C", "k", 0x15));
        out.push(item("C", "kh", 0x16));
        continue;
      }
    }
    if (it.kind === "C" && it.off === 0x2F && it.phone === "y") {
      if ((lang === "bn" || lang === "as") && (!afterCons || prev.phone === "r")) it = item("C", lang === "bn" ? "j" : "z", it.off);
      else if (lang === "or" && !afterCons) it = item("C", "j", it.off);
    }
    out.push(it);
  }
  return out;
}

function tamilVoicing(items) {
  const NATIVE = { 0x15: ["k", "g"], 0x1A: ["s", "j"], 0x1F: ["tt", "dd"], 0x24: ["t", "d"], 0x2A: ["p", "b"] };
  const seq = items.filter((it) => ["C", "V", "@"].includes(it.kind) && !it.deleted);
  seq.forEach((it, idx) => {
    if (it.kind !== "C") return;
    const prev = idx > 0 ? seq[idx - 1] : null;
    const nxt = idx + 1 < seq.length ? seq[idx + 1] : null;
    if (it.off === 0x31) {
      if ((nxt && nxt.kind === "C" && nxt.off === 0x31) || (prev && prev.kind === "C" && prev.off === 0x31)) it.phone = "tx";
      return;
    }
    if (!has(NATIVE, it.off)) return;
    const [voiceless, voiced] = NATIVE[it.off];
    const gem = (prev && prev.kind === "C" && prev.off === it.off) || (nxt && nxt.kind === "C" && nxt.off === it.off);
    if (prev === null) it.phone = voiceless;
    else if (gem) it.phone = it.off === 0x1A ? "c" : voiceless;
    else if (nxt && nxt.kind === "C") it.phone = it.off === 0x1A ? "c" : it.off === 0x15 ? "k" : voiceless;
    else if (prev.kind === "C" && NASALS.has(prev.phone)) it.phone = voiced;
    else if (prev.kind === "C" && VOICELESS_OBSTRUENTS.has(prev.phone)) it.phone = voiceless;
    else it.phone = it.off === 0x1A ? "s" : voiced;
  });
}

function malayalamRules(items) {
  const seq = items.filter((it) => ["C", "V", "@"].includes(it.kind) && !it.deleted);
  seq.forEach((it, idx) => {
    if (it.kind !== "C") return;
    const prev = idx > 0 ? seq[idx - 1] : null;
    const nxt = idx + 1 < seq.length ? seq[idx + 1] : null;
    if (it.off === 0x28) {
      if (prev === null || (nxt && nxt.kind === "C" && ["t", "th", "d", "dh"].includes(nxt.phone))) it.phone = "n";
      else it.phone = "nx";
    } else if (it.off === 0x31) {
      if (prev && prev.kind === "C" && (prev.off === 0x28 || prev.off === 0x7B) && (prev.phone === "n" || prev.phone === "nx")) {
        it.phone = "d";
        prev.phone = "nx";
      } else if ((nxt && nxt.kind === "C" && nxt.off === 0x31) || (prev && prev.kind === "C" && prev.off === 0x31)) {
        it.phone = "tx";
      } else {
        it.phone = "rx";
      }
    }
  });
}

function resolve(items, lang) {
  const cfg = LANG[lang];
  const flat = [];
  const pending = new Map();
  for (const it of items) {
    if (it.kind === "C") flat.push(it.phone);
    else if (it.kind === "V") flat.push(...vowelPhones(it.phone, cfg));
    else if (it.kind === "@") { if (!it.deleted) flat.push(cfg.inherent); }
    else if (it.kind === "NAS") { pending.set(flat.length, it.phone); flat.push("<NAS>"); }
    else if (it.kind === "GEM") flat.push("<GEM>");
    else if (it.kind === "VIS") flat.push("<VIS>");
    else if (it.kind === "AYTHAM") flat.push("<AYT>");
    else if (it.kind === "HALEND") flat.push("ix");
  }
  const out = [];
  const last = () => out[out.length - 1];
  flat.forEach((ph, pos) => {
    let nxt = null;
    for (let q = pos + 1; q < flat.length; q++) if (!flat[q].startsWith("<")) { nxt = flat[q]; break; }
    if (ph === "<NAS>") {
      const kind = pending.get(pos);
      const vowelBefore = out.length > 0 && IN_VOWEL_SET.has(last()) && last() !== "ix";
      const mode = cfg.anusvara;
      if (kind === "candrabindu" && vowelBefore) out[out.length - 1] = nasalize(last());
      else if (mode === "velar" && kind === "anusvara") out.push("ng");
      else if (nxt !== null && has(IN_STOPS_BY_PLACE, nxt)) out.push(IN_STOPS_BY_PLACE[nxt]);
      else if (nxt !== null && NASALS.has(nxt)) out.push(nxt === "nx" ? "n" : nxt);
      else if (mode === "dravidian") out.push("m");
      else if (vowelBefore) out[out.length - 1] = nasalize(last());
      else if (kind === "anusvara") out.push("m");
    } else if (ph === "<GEM>") {
      if (nxt !== null && !IN_VOWEL_SET.has(nxt)) out.push(ASPIRATE_TO_PLAIN[nxt] ?? nxt);
    } else if (ph === "<VIS>") {
      out.push("h");
      if (lang === "sa" && nxt === null && out.length >= 2 && IN_VOWEL_SET.has(out[out.length - 2])) {
        const echo = { aa: "a", ii: "i", uu: "u", ee: "e", oo: "o", ai: "i", au: "u" };
        out.push(echo[out[out.length - 2]] ?? out[out.length - 2]);
      }
    } else if (ph === "<AYT>") {
      out.push(nxt === "p" ? "<F>" : "h");
    } else {
      if (out.length && last() === "<F>" && ph === "p") { out[out.length - 1] = "f"; return; }
      out.push(ph);
    }
  });
  return out.filter((p) => p !== null && p !== "<F>");
}

function post(phones, lang) {
  const cfg = LANG[lang];
  const ov = OVERRIDES[lang] || {};
  let res = phones.map((p, idx) => {
    const nxt = idx + 1 < phones.length ? phones[idx + 1] : null;
    if (lang === "bn" && (p === "sh" || p === "ssh" || p === "s")) return SIBILANT_CLUSTER_NEXT.has(nxt) ? "s" : "sh";
    if (lang === "as" && (p === "sh" || p === "ssh" || p === "s")) return SIBILANT_CLUSTER_NEXT.has(nxt) ? "s" : "x";
    return ov[p] ?? p;
  });
  if (cfg.mergeLen) {
    const m = { ii: "i", uu: "u", "ii~": "i~", "uu~": "u~" };
    if (cfg.mergeLen === "all") Object.assign(m, { ee: "e", oo: "o", "ee~": "e~", "oo~": "o~" });
    res = res.map((p) => m[p] ?? p);
  }
  return res;
}

/** Bare phone names for one word in a Brahmic script (brahmic.word_to_phones), or null. */
export function wordToPhones(word, lang) {
  word = word.normalize("NFC");
  const script = scriptOfWord(word);
  if (!has(BRAHMIC_BASE, script ?? "")) return null;
  let items = parse(word, script, lang);
  if (!items || !items.length) return null;
  items = clusterRules(items, lang);
  deleteSchwas(items, LANG[lang].schwa, lang === "bn" || lang === "as");
  if (lang === "ta") tamilVoicing(items);
  else if (lang === "ml") malayalamRules(items);
  const phones = post(resolve(items, lang), lang);
  return phones.length ? phones : null;
}

/** "in_*" symbols for one Indic word (frontend.indic_word), or null if it cannot be read. */
export function indicWord(word, lang) {
  const script = scriptOfWord(word);
  if (script === null || !has(BRAHMIC_BASE, script)) return null;  // mixed / Perso-Arabic / Ol Chiki / Meetei
  let rules = langScripts(lang).includes(script) ? lang : SCRIPT_DEFAULT_LANG[script];
  if (!has(LANG, rules) || LANG[rules].inherent === null) rules = SCRIPT_DEFAULT_LANG[script];
  const phones = wordToPhones(word, rules);
  return phones ? phones.map((p) => "in_" + p) : null;
}

// ------------------------------------------------------------------ text normalisation (frontend.py)
// Python's \s (str.isspace) set, as in frontend.js
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PUNCT_MAP = {
  "।": ".", "॥": ".", "|": ".", "…": ".", "۔": ".", "؟": "?", "،": ",", "؛": ";",
  "᱾": ".", "᱿": ".", "꯫": ".", "꫰": ".", "꫱": ".",
  "—": ",", "–": ",", "‘": " ", "’": " ", "“": " ", "”": " ", '"': " ", "'": " ",
  "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ", "/": " ", "\\": " ",
  "-": " ", "_": " ", "*": " ", "«": " ", "»": " ", "<": " ", ">": " ", "~": " ",
  "‌": "", " ": " ", "​": "", "﻿": "",
};
const ML_CHILLU = [["ണ്‍", "ൺ"], ["ന്‍", "ൻ"], ["ര്‍", "ർ"], ["ല്‍", "ൽ"], ["ള്‍", "ൾ"], ["ക്‍", "ൿ"]];
const INDIC_CLASS = "\\u0900-\\u0d7f\\u1c50-\\u1c7f\\u1cd0-\\u1cff\\ua8e0-\\ua8ff\\uaae0-\\uaaff\\uabc0-\\uabff"
  + "\\u0600-\\u06ff\\u0750-\\u077f\\u08a0-\\u08ff\\ufb50-\\ufdff\\ufe70-\\ufeff\\u200d";
const R = {
  spaceBeforePunct: new RegExp(`[${WS}]*([.,?!;:])`, "gu"),
  punctRun: new RegExp(`([.,?!;:])(?:[${WS}]*[.,?!;:])+`, "gu"),
  wsRun: new RegExp(`[${WS}]+`, "gu"),
  trim: new RegExp(`^[${WS}]+|[${WS}]+$`, "gu"),
  token: new RegExp(`[${INDIC_CLASS}]+|[A-Za-z]+|[.,?!;:]|[^${WS}]`, "gu"),
  indicWord: new RegExp(`^[${INDIC_CLASS}]+$`, "u"),
  latinWord: /^[A-Za-z]+$/u,
  enToken: /[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]/gu,
};

/** frontend.normalize */
export function normalize(text) {
  text = text.normalize("NFC");
  for (const [a, b] of ML_CHILLU) text = text.replaceAll(a, b);
  text = Array.from(text.replaceAll(ZWJ, ""), (c) => (has(PUNCT_MAP, c) ? PUNCT_MAP[c] : c)).join("");
  text = text.replace(R.spaceBeforePunct, "$1");
  text = text.replace(R.punctRun, "$1");
  return text.replace(R.wsRun, " ").replace(R.trim, "");
}

// ------------------------------------------------------------------ numbers for the web app
// Training text had no digits. At runtime: Kannada uses its own number reader (frontend.js),
// English the Indian-English one, the 10 other languages numbers_multi.js. Numbers in English
// words remain only as a fallback for a language without number data.
const INDIC_DIGITS = {};
for (const base of Object.values(BRAHMIC_BASE)) for (let d = 0; d < 10; d++) INDIC_DIGITS[String.fromCodePoint(base + 0x66 + d)] = String(d);
const toAsciiDigits = (t) => Array.from(t, (c) => INDIC_DIGITS[c] ?? c).join("");
const DIGIT_WORDS = "zero one two three four five six seven eight nine".split(" ");
const digitsEn = (s) => Array.from(s, (d) => DIGIT_WORDS[+d]).join(" ");
const numberEn = (raw) => {
  const [whole, frac] = raw.replaceAll(",", "").split(".");
  return indianEnglishCardinal(whole) + (frac ? " point " + digitsEn(frac) : "");
};
const R_NUM = {
  mobile: /(\+91[\s-]?)?(?<![0-9])([6-9][0-9]{4})[\s-]?([0-9]{5})(?![0-9])/gu,
  money: /(?:₹|\bRs\.?|\bINR)\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)/gu,
  clock: /(?<![0-9])([01]?[0-9]|2[0-3]):([0-5][0-9])(?![0-9])/gu,
  percent: /([0-9]+(?:\.[0-9]+)?)\s*%/gu,
  longDigits: /(?<![0-9,.])([0-9]{7,})(?![0-9,.])/gu,
  number: /[0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?/gu,
};
function numbersInEnglishWords(text) {
  text = toAsciiDigits(text);
  text = text.replace(R_NUM.mobile, (_, cc, a, b) => (cc ? " plus nine one " : " ") + digitsEn(a + b) + " ");
  text = text.replace(R_NUM.money, (_, amount) => {
    const [r, p] = amount.replaceAll(",", "").split(".");
    let words = indianEnglishCardinal(r) + (BigInt(r) === 1n ? " rupee" : " rupees");
    const paise = p ? Number(p.padEnd(2, "0").slice(0, 2)) : 0;
    if (paise) words += " and " + cardinal(paise) + " paise";
    return " " + words + " ";
  });
  text = text.replace(R_NUM.clock, (_, h, m) => " " + cardinal(+h) + (+m === 0 ? "" : +m < 10 ? " oh " + cardinal(+m) : " " + cardinal(+m)) + " ");
  text = text.replace(R_NUM.percent, (_, n) => " " + numberEn(n) + " percent ");
  text = text.replace(R_NUM.longDigits, (_, d) => " " + digitsEn(d) + " ");
  text = text.replace(R_NUM.number, (m) => " " + numberEn(m) + " ");
  return text;
}
export const NUMBERS_IN_ENGLISH = (lang) => lang !== "kn" && lang !== "en" && !NUMBER_LANGS.includes(lang);

// Indian-English words missing from CMU, used in number reading (same as frontend.js)
const EXTRA_LEXICON = {
  lakh: ["L", "AA1", "K"], lakhs: ["L", "AA1", "K", "S"], crore: ["K", "R", "AO1", "R"], crores: ["K", "R", "AO1", "R", "Z"],
  rupee: ["R", "UW1", "P", "IY0"], rupees: ["R", "UW1", "P", "IY0", "Z"], paise: ["P", "AY1", "S", "EY0"], paisa: ["P", "AY1", "S", "AA0"],
  gigabit: ["G", "IH1", "G", "AH0", "B", "IH2", "T"], gigabits: ["G", "IH1", "G", "AH0", "B", "IH2", "T", "S"],
  gigahertz: ["G", "IH1", "G", "AH0", "HH", "ER2", "T", "S"], kilohertz: ["K", "IH1", "L", "AH0", "HH", "ER2", "T", "S"],
  megabits: ["M", "EH1", "G", "AH0", "B", "IH2", "T", "S"],
  megapixel: ["M", "EH1", "G", "AH0", "P", "IH2", "K", "S", "AH0", "L"],
  megapixels: ["M", "EH1", "G", "AH0", "P", "IH2", "K", "S", "AH0", "L", "Z"],
  milliamp: ["M", "IH1", "L", "IY0", "AE2", "M", "P"],
  covid: ["K", "OW1", "V", "IH0", "D"], isro: ["IH1", "S", "R", "OW0"], saarc: ["S", "AA1", "R", "K"],
  sebi: ["S", "EH1", "B", "IY0"], neet: ["N", "IY1", "T"],
};

function englishWord(word, unknown) {
  const w = word.toLowerCase();
  let phones = lookup(w);
  if (phones === undefined && has(EXTRA_LEXICON, w)) phones = EXTRA_LEXICON[w];
  if (phones === undefined) {
    unknown.push(w);
    phones = romanizedArpabet(w);               // Indian spelling rules (as in PiVo-KN)
  }
  return phones;
}

// ------------------------------------------------------------------ token assembly (frontend._assemble)
function assemble(items) {
  const tokens = ["<bos>"];
  let words = 0;
  for (const [kind, value, symbols] of items) {
    const lastTok = tokens[tokens.length - 1];
    if (kind === "punct") {
      if (lastTok === "<bos>" || SPECIAL.slice(4).includes(lastTok)) continue;
      if (lastTok === "<space>") tokens.pop();
      tokens.push(value);
      continue;
    }
    if (lastTok !== "<bos>" && lastTok !== "<space>") tokens.push("<space>");
    tokens.push(...symbols);
    words++;
  }
  tokens.push("<eos>");
  return { tokens, words };
}

/**
 * Text in `lang` -> { normalized, symbols, ids, unknown, skipped, lang }.
 * numbers: expand digits (web app). With numbers=false and clean text this is exactly the
 * training frontend (frontend.phonemize, strict text).
 */
export function phonemize(text, lang, { numbers = true } = {}) {
  if (!has(LANG, lang)) throw new Error(`unknown language ${lang}`);
  const skipped = [], unknown = [], items = [];
  let normalized;
  if (lang === "en") {
    normalized = normalizeEnglish(numbers ? normalizeIndianEnglish(text) : text, { lenient: true, skipped });
    for (const [tok] of normalized.matchAll(R.enToken)) {
      if (SPECIAL.includes(tok)) items.push(["punct", tok, null]);
      else items.push(["word", tok.toLowerCase(), englishWord(tok, unknown)]);
    }
  } else {
    if (numbers) {
      text = lang === "kn" ? normalizeKannadaText(text)
        : NUMBER_LANGS.includes(lang) ? normalizeNumbers(text, lang) : numbersInEnglishWords(text);
    }
    normalized = normalize(text);
    for (const [tok] of normalized.matchAll(R.token)) {
      if (SPECIAL.includes(tok)) { items.push(["punct", tok, null]); continue; }
      if (R.latinWord.test(tok)) { items.push(["word", tok.toLowerCase(), englishWord(tok, unknown)]); continue; }
      if (R.indicWord.test(tok)) {
        const w = tok.replace(/^‍+|‍+$/gu, "");
        if (!w) continue;
        const symbols = indicWord(w, lang);
        if (symbols) items.push(["word", w, symbols]);
        else skipped.push(w);
        continue;
      }
      skipped.push(tok);
    }
  }
  const { tokens, words } = assemble(items);
  if (!words) return { normalized, symbols: ["<bos>", "<eos>"], ids: [1, 2], unknown, skipped, lang };
  return { normalized, symbols: tokens, ids: tokens.map((s) => SYMBOL_TO_ID.get(s)), unknown, skipped, lang };
}

/** True when phonemizing this text may need the English dictionary. */
export const lexiconNeeded = (text, lang) => lang === "en" || /[A-Za-z]/.test(text) || (NUMBERS_IN_ENGLISH(lang) && /[0-9०-९০-৯੦-੯૦-૯୦-୯௦-௯౦-౯೦-೯൦-൯]/.test(text));

/** Main script of a text ("latin", "devanagari", ...) by letter count, or null. */
export function mainScript(text) {
  const count = {};
  for (const ch of text) {
    const s = scriptOfChar(ch);
    if (s) count[s] = (count[s] || 0) + 1;
  }
  let best = null;
  for (const [s, n] of Object.entries(count)) if (!best || n > count[best]) best = s;
  return best;
}
