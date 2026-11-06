// PiVo-KN text frontend for the browser (metadata frontend "pivo-kn-en-v1").
//
// A line-by-line port of src/pivo/text.py, src/pivo/normalize_kn.py and src/pivo/kannada.py.
// It must produce the same token ids as Python: web/tests/parity.mjs checks that on a large
// sentence set. When you change a rule in Python, change it here too and rerun the parity test.
//
// Python regex details reproduced here:
//   \d  -> [0-9]           (Kannada digits are translated to ASCII first, as in Python)
//   \s  -> Python's str.isspace() set (differs slightly from JS \s)
//   \b  -> (?<![\p{L}\p{N}_]) / (?![\p{L}\p{N}_])   (Python \b is Unicode-aware, JS \b is not)

// ------------------------------------------------------------------ regex helpers
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WORD = "[\\p{L}\\p{N}_]";
const B0 = `(?<!${WORD})`; // \b where the next character is a word character
const B1 = `(?!${WORD})`;  // \b where the previous character is a word character
const KN = "\\u0C80-\\u0CFF";

/** Python-flavoured pattern -> RegExp: \d \D \s \S follow Python's meaning. */
function rx(src, flags = "gu") {
  let out = "", inClass = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (c === "\\") {
      const n = src[++i];
      if (inClass && n === "d") out += "0-9";
      else if (inClass && n === "s") out += WS;
      else if (!inClass && n === "d") out += "[0-9]";
      else if (!inClass && n === "D") out += "[^0-9]";
      else if (!inClass && n === "s") out += `[${WS}]`;
      else if (!inClass && n === "S") out += `[^${WS}]`;
      else out += "\\" + n;
      continue;
    }
    if (c === "[" && !inClass) inClass = true;
    else if (c === "]" && inClass) inClass = false;
    out += c;
  }
  return new RegExp(out, flags);
}

/** re.sub with a function receiving the match (m[1], m.groups.name, ...). */
function sub(re, fn, text) {
  let out = "", last = 0;
  for (const m of text.matchAll(re)) {
    out += text.slice(last, m.index) + fn(m);
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const WS_RUN = rx(String.raw`\s+`);
const collapse = (t) => t.replace(WS_RUN, " ").replace(rx(String.raw`^\s+|\s+$`), "");
const translate = (text, map) => Array.from(text, (c) => (c in map ? map[c] : c)).join("");
const byLengthDesc = (keys) => [...keys].sort((a, b) => b.length - a.length); // stable, like Python
const int = (s) => BigInt(s);

export class FrontendError extends Error {
  constructor(kind, message) { super(message); this.kind = kind; }
}

// ================================================================== English (text.py)
const CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(" ");
const VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(" ");
export const SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"];
const ENGLISH_SYMBOLS = [...SPECIAL, ...CONSONANTS, ...VOWELS.flatMap((v) => [v + "0", v + "1", v + "2"])];
const ENGLISH_TO_ID = new Map(ENGLISH_SYMBOLS.map((s, i) => [s, i]));
const ONES_EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(" ");
const TENS_EN = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split(" ");
const ORDINALS_EN = { one: "first", two: "second", three: "third", five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };
const MONTHS_EN = "January February March April May June July August September October November December".split(" ");
const SCALES_EN = [[10n ** 12n, "trillion"], [10n ** 9n, "billion"], [10n ** 6n, "million"], [1000n, "thousand"], [100n, "hundred"]];

export function cardinal(n) {
  n = BigInt(n);
  if (n < 0n) return "minus " + cardinal(-n);
  if (n < 20n) return ONES_EN[Number(n)];
  if (n < 100n) return TENS_EN[Number(n / 10n)] + (n % 10n ? " " + cardinal(n % 10n) : "");
  for (const [scale, label] of SCALES_EN) {
    if (n >= scale) {
      if (n >= 10n ** 15n) return Array.from(n.toString(), (d) => ONES_EN[+d]).join(" ");
      return cardinal(n / scale) + " " + label + (n % scale ? " " + cardinal(n % scale) : "");
    }
  }
  throw new Error("unreachable");
}

export function ordinal(n) {
  const words = cardinal(n).split(" ");
  const last = words[words.length - 1];
  words[words.length - 1] = ORDINALS_EN[last] ?? (last.endsWith("y") ? last.slice(0, -1) + "ieth" : last + "th");
  return words.join(" ");
}

function validDate(y, m, d) {
  if (y < 1 || y > 9999 || m < 1 || m > 12 || d < 1) return false;
  const days = [31, (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0 ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return d <= days[m - 1];
}

const EN_PUNCT = { "’": "'", "‘": "'", "“": '"', "”": '"', "—": ",", "–": ",", "…": "..." };
const EN_ABBREVIATIONS = { Mr: "Mister", Mrs: "Missus", Dr: "Doctor", Prof: "Professor", St: "Saint", vs: "versus", etc: "et cetera" };
const R_EN = {
  iso: rx(String.raw`${B0}(\d{4})-(\d{2})-(\d{2})${B1}`),
  slash: rx(String.raw`\d/\d`, "u"),
  slashLenient: rx(String.raw`(\d)\s*/\s*(\d)`),
  abbr: Object.entries(EN_ABBREVIATIONS).map(([a, e]) => [rx(String.raw`${B0}${a}\.(?=\s|$)`, "giu"), e]),
  badMoney: rx(String.raw`[$£€]\d[\d,]*\.\d{3,}`, "u"),
  money: rx(String.raw`([$£€])(\d[\d,]*(?:\.\d{1,2})?)(?!\d)`),
  ordinal: rx(String.raw`${B0}(\d+)(?:st|nd|rd|th)${B1}`, "giu"),
  grouped: rx(String.raw`${B0}\d{1,3}(?:,\d{3})+${B1}`),
  number: rx(String.raw`-?(?:\d+\.\d+|\d+)`),
  nasa: rx(String.raw`${B0}NASA${B1}`),
  initialism: rx(String.raw`${B0}[A-Z]{2,5}${B1}`),
  brackets: /[()[\]{}"/]/gu,
  unsupported: rx(String.raw`[^a-zA-Z\s'.,?!;:]`),
  word: /[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]/gu,
};

/** text.normalize. lenient: never throws; drops what it cannot read and lists it in `skipped`. */
export function normalizeEnglish(text, { lenient = false, skipped = [] } = {}) {
  text = translate(text.normalize("NFKC"), EN_PUNCT);
  text = sub(R_EN.iso, (m) => {
    const [y, mo, d] = [+m[1], +m[2], +m[3]];
    if (!validDate(y, mo, d)) {
      if (lenient) return m[0];
      throw new FrontendError("date", `invalid date ${m[0]}`);
    }
    return `${MONTHS_EN[mo - 1]} ${ordinal(d)}, ${cardinal(y)}`;
  }, text);
  if (R_EN.slash.test(text)) {
    if (!lenient) throw new FrontendError("slash", "write slash dates/fractions in words, or use an ISO date");
    text = sub(R_EN.slashLenient, (m) => `${m[1]} by ${m[2]}`, text);
  }
  for (const [re, expansion] of R_EN.abbr) text = text.replace(re, expansion);
  if (R_EN.badMoney.test(text) && !lenient) {
    throw new FrontendError("currency", "currency amounts must have at most two decimal places");
  }
  text = sub(R_EN.money, (m) => {
    const raw = m[2].replaceAll(",", "");
    const dot = raw.indexOf(".");
    const whole = dot < 0 ? raw : raw.slice(0, dot), fraction = dot < 0 ? "" : raw.slice(dot + 1);
    const symbol = m[1];
    const unit = { "$": "dollar", "£": "pound", "€": "euro" }[symbol];
    let result = cardinal(whole) + " " + unit + (int(whole) !== 1n ? "s" : "");
    const cents = fraction ? Number(fraction.padEnd(2, "0")) : 0;
    if (cents) {
      result += " and " + cardinal(cents) + (cents === 1 && symbol === "£" ? " penny" : symbol === "£" ? " pence"
        : cents === 1 ? " cent" : " cents");
    }
    return result;
  }, text);
  text = sub(R_EN.ordinal, (m) => ordinal(m[1]), text);
  text = sub(R_EN.grouped, (m) => m[0].replaceAll(",", ""), text);
  text = sub(R_EN.number, (m) => {
    const s = m[0], neg = s.startsWith("-"), body = s.replace(/^-+/, "");
    let words;
    if (s.includes(".")) {
      const [whole, frac] = [body.split(".")[0], s.split(".")[1]];
      words = [cardinal(whole), Array.from(frac, (c) => ONES_EN[+c]).join(" ")].join(" point ");
    } else {
      words = cardinal(body);
    }
    return (neg ? "minus " : "") + words;
  }, text);
  text = text.replaceAll("%", " percent ").replaceAll("&", " and ").replaceAll("+", " plus ");
  text = text.replace(R_EN.nasa, "nasa");
  text = sub(R_EN.initialism, (m) => Array.from(m[0]).join(" "), text);
  text = text.replaceAll("-", " ").replace(R_EN.brackets, " ");
  const bad = text.match(R_EN.unsupported);
  if (bad) {
    if (!lenient) {
      throw new FrontendError("unsupported", `unsupported text characters after normalization: ${[...new Set(bad)].sort().join(" ")}`);
    }
    skipped.push(...bad);
    text = text.replace(R_EN.unsupported, " ");
  }
  return collapse(text);
}

// ------------------------------------------------------------------ CMU dictionary (lazy)
let LEXICON = null;

/** Load the lexicon text: one "word PH ON ES" line per word, first CMU pronunciation only. */
export function setLexicon(raw) {
  const map = new Map();
  for (const line of raw.split("\n")) {
    const cut = line.indexOf(" ");
    if (cut > 0) map.set(line.slice(0, cut), line.slice(cut + 1).trim());
  }
  LEXICON = map;
}
export const lexiconLoaded = () => LEXICON !== null;
/** True when the text can reach a dictionary lookup (English path, or Latin words in Kannada). */
export const lexiconNeeded = (text) => /[A-Za-z]/.test(text) || !HAS_KANNADA.test(text);

export function lookup(word) {
  if (!LEXICON) throw new FrontendError("lexicon", "the English dictionary is not loaded yet");
  const v = LEXICON.get(word);
  return v === undefined ? undefined : v.split(" ");
}
const spell = (word) => Array.from(word).filter((c) => /[a-z]/.test(c)).flatMap((c) => lookup(c));

function phonemizeEnglish(normalized, { strict, overrides }) {
  const over = new Map(Object.entries(overrides).map(([k, v]) => [k.toLowerCase(), v]));
  const tokens = ["<bos>"], unknown = [];
  for (const [item] of normalized.matchAll(R_EN.word)) {
    if (SPECIAL.includes(item)) { tokens.push(item); continue; }
    const word = item.toLowerCase();
    let phones = over.get(word);
    if (phones === undefined) phones = lookup(word);
    if (phones === undefined) { unknown.push(word); phones = spell(word); }
    if (tokens[tokens.length - 1] !== "<bos>" && tokens[tokens.length - 1] !== "<space>") tokens.push("<space>");
    for (const p of phones) {
      if (!ENGLISH_TO_ID.has(p)) throw new FrontendError("phone", `unknown ARPAbet phone ${p} in ${word}`);
      tokens.push(p);
    }
  }
  if (unknown.length && strict) {
    throw new FrontendError("oov", "out-of-vocabulary words require pronunciation overrides: " + [...new Set(unknown)].sort().join(", "));
  }
  tokens.push("<eos>");
  return { tokens, unknown };
}

// ================================================================== Kannada numbers (normalize_kn.py)
const ONES = ["ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
  "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು",
  "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು"];
const TENS = { 2: "ಇಪ್ಪತ್ತು", 3: "ಮೂವತ್ತು", 4: "ನಲವತ್ತು", 5: "ಐವತ್ತು", 6: "ಅರವತ್ತು", 7: "ಎಪ್ಪತ್ತು", 8: "ಎಂಬತ್ತು", 9: "ತೊಂಬತ್ತು" };
const HUNDREDS = { 1: "ನೂರು", 2: "ಇನ್ನೂರು", 3: "ಮುನ್ನೂರು", 4: "ನಾನೂರು", 5: "ಐನೂರು", 6: "ಆರುನೂರು", 7: "ಏಳುನೂರು", 8: "ಎಂಟುನೂರು", 9: "ಒಂಬೈನೂರು" };
const THOUSAND = "ಸಾವಿರ", LAKH = "ಲಕ್ಷ", CRORE = "ಕೋಟಿ";
const POINT = "ಪಾಯಿಂಟ್", PERCENT = "ಶೇಕಡಾ", RUPEE = "ರೂಪಾಯಿ", PAISE = "ಪೈಸೆ", HOUR = "ಗಂಟೆ", MINUTE = "ನಿಮಿಷ";
const MONTHS = ["ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"];
const U = "ು";
const VOWEL_SIGN = { "ಅ": "", "ಆ": "ಾ", "ಇ": "ಿ", "ಈ": "ೀ", "ಉ": "ು", "ಊ": "ೂ", "ಎ": "ೆ", "ಏ": "ೇ", "ಐ": "ೈ", "ಒ": "ೊ", "ಓ": "ೋ", "ಔ": "ೌ" };
const FRONT_VOWEL_END = ["ಿ", "ೀ", "ೆ", "ೇ", "ೈ"];

const stem = (w) => (w.endsWith(U) ? w.slice(0, -1) : w);
function join(left, right) {
  const s = stem(left);
  return right[0] in VOWEL_SIGN ? s + VOWEL_SIGN[right[0]] + right.slice(1) : s + right;
}
function below100(n) {
  if (n < 20) return ONES[n];
  const t = Math.floor(n / 10), o = n % 10;
  return o === 0 ? TENS[t] : join(TENS[t], ONES[o]);
}
function below1000(n) {
  const h = Math.floor(n / 100), rest = n % 100;
  if (!h) return below100(rest);
  return rest === 0 ? HUNDREDS[h] : stem(HUNDREDS[h]) + " " + below100(rest);
}

export function kannadaCardinal(n) {
  n = BigInt(n);
  if (n < 0n) return "ಮೈನಸ್ " + kannadaCardinal(-n);
  if (n === 0n) return ONES[0];
  const crore = n / 10n ** 7n; n %= 10n ** 7n;
  const lakh = Number(n / 10n ** 5n); n %= 10n ** 5n;
  const thousand = Number(n / 1000n), rest = Number(n % 1000n);
  const parts = [];
  if (crore) parts.push([kannadaCardinal(crore), CRORE]);
  if (lakh) parts.push([below100(lakh), LAKH]);
  if (thousand === 1 && rest && !parts.length) parts.push([THOUSAND]);
  else if (thousand) parts.push([below100(thousand), THOUSAND]);
  if (rest) parts.push([below1000(rest)]);
  const words = [];
  parts.forEach((part, i) => {
    const last = i === parts.length - 1;
    const tail = part[part.length - 1];
    if (!last && (tail === THOUSAND || tail === LAKH)) part = [...part.slice(0, -1), tail + "ದ"];
    words.push(...part);
  });
  return words.join(" ");
}

const digitsKn = (s) => Array.from(s, (d) => ONES[+d]).join(" ");

export function attachSuffix(words, suffix) {
  if (!suffix) return words;
  const cut = words.lastIndexOf(" ");
  const head = cut < 0 ? "" : words.slice(0, cut);
  let last = words.slice(cut + 1);
  if (last.endsWith(U) && suffix.startsWith("ಗ")) suffix = "ಕ್ಕ" + suffix.slice(1);
  if (FRONT_VOWEL_END.some((v) => last.endsWith(v))) {
    if (suffix.startsWith("ರ")) suffix = "ಯ" + suffix.slice(1);
    else if (suffix.startsWith("ಕ್ಕ")) suffix = "ಗ" + suffix.slice(3);
  }
  last = suffix[0] in VOWEL_SIGN ? join(last, suffix) : stem(last) + suffix;
  return head ? head + " " + last : last;
}

function kannadaNumberText(raw) {
  raw = raw.replaceAll(",", "");
  const dot = raw.indexOf(".");
  const whole = dot < 0 ? raw : raw.slice(0, dot), frac = dot < 0 ? "" : raw.slice(dot + 1);
  let words = whole ? kannadaCardinal(whole) : ONES[0];
  if (frac) words += " " + POINT + " " + digitsKn(frac);
  return words;
}

// ================================================================== Kannada rules
const KN_DIGITS = Object.fromEntries(Array.from("೦೧೨೩೪೫೬೭೮೯", (c, i) => [c, String(i)]));
const NUM = String.raw`\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?`;
const SUF = `(?<suf>[${KN}\\u200c\\u200d]*)`;
const END = String.raw`(?!\d)`;

const ABBREVIATIONS = {
  "ಡಾ.": "ಡಾಕ್ಟರ್", "ಪ್ರೊ.": "ಪ್ರೊಫೆಸರ್", "ಶ್ರೀ.": "ಶ್ರೀ", "ಕ್ರಿ.ಶ.": "ಕ್ರಿಸ್ತ ಶಕ",
  "ಕ್ರಿ.ಪೂ.": "ಕ್ರಿಸ್ತ ಪೂರ್ವ", "ರೂ.": RUPEE, "ಶೇ.": PERCENT, "ನಂ.": "ನಂಬರ್",
  "ಉದಾ.": "ಉದಾಹರಣೆಗೆ", "ಕಿ.ಮೀ.": "ಕಿಲೋಮೀಟರ್", "ಕಿ.ಮೀ": "ಕಿಲೋಮೀಟರ್",
  "ಸೆಂ.ಮೀ.": "ಸೆಂಟಿಮೀಟರ್", "ಮಿ.ಮೀ.": "ಮಿಲಿಮೀಟರ್", "ಕೆ.ಜಿ.": "ಕೆಜಿ", "ಕೆ.ಜಿ": "ಕೆಜಿ",
};
const UNITS = {
  "km/h": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಗಂಟೆ", "kmph": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಗಂಟೆ", "km": "ಕಿಲೋಮೀಟರ್",
  "kg": "ಕಿಲೋಗ್ರಾಂ", "cm": "ಸೆಂಟಿಮೀಟರ್", "mm": "ಮಿಲಿಮೀಟರ್", "ml": "ಮಿಲಿಲೀಟರ್",
  "gm": "ಗ್ರಾಂ", "g": "ಗ್ರಾಂ", "ltr": "ಲೀಟರ್", "l": "ಲೀಟರ್", "m": "ಮೀಟರ್",
  "°C": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್", "°c": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್", "°": "ಡಿಗ್ರಿ", "GB": "ಜಿಬಿ", "MB": "ಎಂಬಿ",
  "hrs": HOUR, "hr": HOUR, "mins": MINUTE, "min": MINUTE,
};
const INITIALS = new Set(["ಎ", "ಬಿ", "ಸಿ", "ಡಿ", "ಇ", "ಎಫ್", "ಜಿ", "ಎಚ್", "ಹೆಚ್", "ಐ", "ಜೆ", "ಕೆ", "ಎಲ್", "ಎಂ",
  "ಎನ್", "ಒ", "ಪಿ", "ಕ್ಯೂ", "ಆರ್", "ಎಸ್", "ಟಿ", "ಯು", "ವಿ", "ಡಬ್ಲ್ಯು", "ಎಕ್ಸ್", "ವೈ", "ಜೆಡ್", "ಝಡ್"]);
const ACRONYM_WORDS = { NASA: "ನಾಸಾ", ISRO: "ಇಸ್ರೋ", COVID: "ಕೋವಿಡ್", UNESCO: "ಯುನೆಸ್ಕೋ", NATO: "ನ್ಯಾಟೋ", WiFi: "ವೈಫೈ", OK: "ಓಕೆ" };
const SYMBOLS_KN = { "&": " ಮತ್ತು ", "+": " ಪ್ಲಸ್ ", "=": " ಸಮ ", "@": " ಅಟ್ ", "#": " ", "~": " ", "|": " ", "•": " ", "·": " ", "^": " ", "<": " ", ">": " " };
const ORPHAN_SUFFIXES = ["ಕ್ಕೆ", "ಕ್ಕೂ", "ಕ್ಕಿಂತ", "ರಿಂದ", "ರಂದು", "ರಲ್ಲಿ", "ರವರೆಗೆ", "ರೊಳಗೆ", "ರಷ್ಟು",
  "ರಂತೆ", "ದಿಂದ", "ದಲ್ಲಿ", "ದವರೆಗೆ", "ಗೆ", "ಗಿಂತ", "ನೇ", "ರ", "ದ"];

function periodWord(hour, ampm) {
  if (ampm === "am") return hour === 12 ? "ಮಧ್ಯರಾತ್ರಿ" : hour <= 3 ? "ರಾತ್ರಿ" : "ಬೆಳಿಗ್ಗೆ";
  if (hour === 12 || hour <= 3) return "ಮಧ್ಯಾಹ್ನ";
  return hour <= 6 ? "ಸಂಜೆ" : "ರಾತ್ರಿ";
}
function timeKn(m) {
  let hour = +m.groups.h; const minute = +m.groups.m;
  let ampm = (m.groups.ampm || "").replaceAll(".", "").toLowerCase() || null;
  if (ampm === null && hour > 12) { ampm = "pm"; hour -= 12; }
  const words = [];
  if (ampm) words.push(periodWord(hour, ampm));
  words.push(kannadaCardinal(hour), HOUR);
  if (minute) words.push(kannadaCardinal(minute), MINUTE);
  return " " + attachSuffix(words.join(" "), m.groups.suf) + " ";
}
function dateKn(day, month, year, suffix) {
  if (!(month >= 1 && month <= 12 && day >= 1 && day <= 31)) return null;
  return " " + attachSuffix(`${kannadaCardinal(day)} ${MONTHS[month - 1]} ${kannadaCardinal(year)}`, suffix) + " ";
}
function moneyKn(m) {
  const amount = m.groups.amt.replaceAll(",", "");
  const dot = amount.indexOf(".");
  const rupees = dot < 0 ? amount : amount.slice(0, dot), paise = dot < 0 ? "" : amount.slice(dot + 1);
  let words = kannadaCardinal(rupees) + " " + RUPEE;
  const p = paise ? Number(paise.padEnd(2, "0").slice(0, 2)) : 0;
  if (p) words += " " + kannadaCardinal(p) + " " + PAISE;
  return " " + attachSuffix(words, m.groups.suf) + " ";
}

const UNIT_ALT = byLengthDesc(Object.keys(UNITS)).map(escapeRe).join("|");
const ORPHAN_ALT = byLengthDesc(ORPHAN_SUFFIXES).join("|");
const R_KN = {
  abbr: byLengthDesc(Object.keys(ABBREVIATIONS)).map((w) => [rx(`(?<![${KN}])${escapeRe(w)}`), " " + ABBREVIATIONS[w] + " "]),
  initials: rx(String.raw`(?<![${KN}])([${KN}]+)\.(?=\s*[${KN}])`),
  orphans: rx(String.raw`(\d|%|[AaPp]\.?[Mm]\.?)\s+(${ORPHAN_ALT})(?![${KN}])`),
  mobile: rx(String.raw`(\+91[\s-]?)?(?<!\d)([6-9]\d{4})[\s-]?(\d{5})${END}${SUF}`),
  landline: rx(String.raw`(?<!\d)0\d{2,4}[\s-]\d{6,8}${END}${SUF}`),
  pin: rx(String.raw`(ಪಿನ್\s?ಕೋಡ್|ಪಿನ್|PIN|[Pp]in\s?[Cc]ode|[${KN}]+\s?-)\s*:?\s*(\d{6})${END}${SUF}`),
  isoDate: rx(String.raw`(?<!\d)(\d{4})-(\d{2})-(\d{2})${END}${SUF}`),
  date: rx(String.raw`(?<![\d.])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})${END}${SUF}`),
  time: rx(String.raw`(?<!\d)(?<h>[01]?\d|2[0-3]):(?<m>[0-5]\d)\s*(?<ampm>[AaPp]\.?[Mm]\.?)?(?![A-Za-z])${SUF}`),
  money: rx(String.raw`(?:₹|${B0}Rs\.?|${B0}INR|${RUPEE})\s*(?<amt>${NUM})${END}${SUF}`),
  she: rx(`(?<![${KN}])ಶೇ(?![${KN}])`),
  percent: rx(String.raw`(${NUM})\s*%${SUF}`),
  range: rx(String.raw`(?<![\d.,])(\d+)\s?[-–]\s?(\d+)${END}${SUF}`),
  units: rx(String.raw`(${NUM})\s?(${UNIT_ALT})(?![A-Za-z])`),
  ordinal: rx(String.raw`(?<!\d)(\d+)(?:st|nd|rd|th)${B1}`),
  longDigits: rx(String.raw`(?<![.,\d])(0\d+|\d{7,})${END}${SUF}`),
  number: rx(String.raw`(${NUM})${SUF}`),
  acronym: rx(String.raw`${B0}[A-Za-z]{2,6}${B1}`),
  spaceBeforePunct: rx(String.raw`\s+([.,?!;:])`),
};

export function normalizeKannadaText(text) {
  text = translate(text.normalize("NFC"), KN_DIGITS);
  for (const [re, spoken] of R_KN.abbr) text = text.replace(re, () => spoken);
  text = sub(R_KN.initials, (m) => (INITIALS.has(m[1]) ? m[1] + " " : m[0]), text);
  text = sub(R_KN.orphans, (m) => m[1] + m[2], text);
  text = sub(R_KN.mobile, (m) => (m[1] ? " ಪ್ಲಸ್ ಒಂಬತ್ತು ಒಂದು " : " ") + attachSuffix(digitsKn(m[2] + m[3]), m.groups.suf) + " ", text);
  text = sub(R_KN.landline, (m) => {
    const suf = m.groups.suf, body = m[0].slice(0, m[0].length - suf.length);
    return " " + attachSuffix(digitsKn(body.replace(/[^0-9]/g, "")), suf) + " ";
  }, text);
  text = sub(R_KN.pin, (m) => m[1].replace(/-+$/, "") + " " + attachSuffix(digitsKn(m[2]), m.groups.suf) + " ", text);
  text = sub(R_KN.isoDate, (m) => dateKn(+m[3], +m[2], +m[1], m.groups.suf) ?? m[0], text);
  text = sub(R_KN.date, (m) => dateKn(+m[1], +m[2], +m[3], m.groups.suf) ?? m[0], text);
  text = sub(R_KN.time, timeKn, text);
  text = sub(R_KN.money, moneyKn, text);
  text = text.replace(R_KN.she, ` ${PERCENT} `);
  text = sub(R_KN.percent, (m) => ` ${PERCENT} ` + attachSuffix(kannadaNumberText(m[1]), m.groups.suf) + " ", text);
  text = sub(R_KN.range, (m) => " " + attachSuffix(kannadaCardinal(m[1]), "ರಿಂದ") + " "
    + attachSuffix(kannadaCardinal(m[2]), m.groups.suf) + " ", text);
  text = sub(R_KN.units, (m) => ` ${kannadaNumberText(m[1])} ${UNITS[m[2]]} `, text);
  text = sub(R_KN.ordinal, (m) => " " + attachSuffix(kannadaCardinal(m[1]), "ನೇ") + " ", text);
  text = sub(R_KN.longDigits, (m) => " " + attachSuffix(digitsKn(m[1]), m.groups.suf) + " ", text);
  text = sub(R_KN.number, (m) => " " + attachSuffix(kannadaNumberText(m[1]), m.groups.suf) + " ", text);
  text = sub(R_KN.acronym, (m) => {
    const w = m[0];
    if (Object.hasOwn(ACRONYM_WORDS, w)) return ACRONYM_WORDS[w];
    if (w === w.toUpperCase() && w.length >= 2 && w.length <= 6) return Array.from(w.toLowerCase()).join(" ");
    return w;
  }, text);
  for (const [symbol, spoken] of Object.entries(SYMBOLS_KN)) text = text.replaceAll(symbol, spoken);
  text = text.replace(R_KN.spaceBeforePunct, "$1");
  return collapse(text);
}

// ================================================================== Indian English
const INDIAN_ENGLISH_LEXICON = {
  lakh: ["L", "AA1", "K"], lakhs: ["L", "AA1", "K", "S"],
  crore: ["K", "R", "AO1", "R"], crores: ["K", "R", "AO1", "R", "Z"],
  rupee: ["R", "UW1", "P", "IY0"], rupees: ["R", "UW1", "P", "IY0", "Z"],
  paise: ["P", "AY1", "S", "EY0"], paisa: ["P", "AY1", "S", "AA0"],
};
const EN_UNITS = { "km/h": "kilometers per hour", "kmph": "kilometers per hour", "km": "kilometers",
  "kg": "kilograms", "cm": "centimeters", "mm": "millimeters", "ml": "milliliters",
  "°C": "degrees celsius", "°c": "degrees celsius", "°": "degrees" };
const DIGIT_WORDS = "zero one two three four five six seven eight nine".split(" ");

export function indianEnglishCardinal(n) {
  n = BigInt(n);
  if (n < 100000n) return cardinal(n);
  const crore = n / 10n ** 7n; n %= 10n ** 7n;
  const lakh = n / 10n ** 5n, rest = n % 10n ** 5n;
  const words = [];
  if (crore) words.push(indianEnglishCardinal(crore) + " crore");
  if (lakh) words.push(cardinal(lakh) + " lakh");
  if (rest) words.push(cardinal(rest));
  return words.join(" ");
}
const enDigits = (s) => Array.from(s, (d) => DIGIT_WORDS[+d]).join(" ");
const R_IE = {
  mobile: rx(String.raw`(\+91[\s-]?)?${B0}([6-9]\d{4})[\s-]?(\d{5})${B1}`),
  date: rx(String.raw`${B0}(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})${B1}`),
  clock: rx(String.raw`${B0}([01]?\d|2[0-3]):([0-5]\d)\s*([AaPp]\.?[Mm]\.?)?(?![A-Za-z])`),
  money: rx(String.raw`(?:₹|${B0}Rs\.?|${B0}INR)\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)`),
  grouped: rx(String.raw`${B0}\d{1,2}(?:,\d{2})+,\d{3}${B1}`),
  units: rx(String.raw`(\d+(?:\.\d+)?)\s?(${byLengthDesc(Object.keys(EN_UNITS)).map(escapeRe).join("|")})(?![A-Za-z])`),
  spaceBeforePunct: rx(String.raw`\s+([.,?!;:])`),
};

export function normalizeIndianEnglish(text) {
  text = text.normalize("NFKC");
  text = sub(R_IE.mobile, (m) => (m[1] ? " plus nine one " : " ") + enDigits(m[2] + m[3]) + " ", text);
  text = sub(R_IE.date, (m) => {
    const day = +m[1], month = +m[2];
    if (!(month >= 1 && month <= 12 && day >= 1 && day <= 31)) return m[0];
    return ` ${ordinal(day)} ${MONTHS_EN[month - 1]}, ${cardinal(m[3])} `;
  }, text);
  text = sub(R_IE.clock, (m) => {
    const hour = +m[1], minute = +m[2];
    const words = cardinal(hour) + (minute === 0 ? " o'clock" : minute < 10 ? " oh " + cardinal(minute) : " " + cardinal(minute));
    const ampm = (m[3] || "").replaceAll(".", "").toUpperCase();
    return " " + words + (ampm ? " " + Array.from(ampm).join(" ") : "") + " ";
  }, text);
  text = sub(R_IE.money, (m) => {
    const amount = m[1].replaceAll(",", "");
    const dot = amount.indexOf(".");
    const rupees = dot < 0 ? amount : amount.slice(0, dot), paise = dot < 0 ? "" : amount.slice(dot + 1);
    const r = BigInt(rupees);
    let words = indianEnglishCardinal(r) + (r === 1n ? " rupee" : " rupees");
    const p = paise ? Number(paise.padEnd(2, "0").slice(0, 2)) : 0;
    if (p) words += " and " + cardinal(p) + " paise";
    return " " + words + " ";
  }, text);
  text = sub(R_IE.grouped, (m) => " " + indianEnglishCardinal(m[0].replaceAll(",", "")) + " ", text);
  text = sub(R_IE.units, (m) => `${m[1]} ${EN_UNITS[m[2]]}`, text);
  text = text.replace(R_IE.spaceBeforePunct, "$1");
  return collapse(text);
}

// ================================================================== Kannada G2P (kannada.py)
const KN_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"];
const KN_CONSONANTS = ["k", "kh", "g", "gh", "ng", "c", "ch", "j", "jh", "ny", "tt", "tth", "dd", "ddh", "nn",
  "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "ll", "sh", "ssh", "s", "h", "f", "z"];
export const SYMBOLS = [...ENGLISH_SYMBOLS, ...[...KN_VOWELS, ...KN_CONSONANTS].map((p) => "kn_" + p)];
export const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]));
export const FRONTEND_NAME = "pivo-kn-en-v1";

const V = { "ಅ": ["a"], "ಆ": ["aa"], "ಇ": ["i"], "ಈ": ["ii"], "ಉ": ["u"], "ಊ": ["uu"],
  "ಋ": ["r", "u"], "ೠ": ["r", "uu"], "ಌ": ["l", "u"], "ೡ": ["l", "uu"],
  "ಎ": ["e"], "ಏ": ["ee"], "ಐ": ["ai"], "ಒ": ["o"], "ಓ": ["oo"], "ಔ": ["au"] };
const M = { "ಾ": ["aa"], "ಿ": ["i"], "ೀ": ["ii"], "ು": ["u"], "ೂ": ["uu"],
  "ೃ": ["r", "u"], "ೄ": ["r", "uu"], "ೢ": ["l", "u"], "ೣ": ["l", "uu"],
  "ೆ": ["e"], "ೇ": ["ee"], "ೈ": ["ai"], "ೊ": ["o"], "ೋ": ["oo"], "ೌ": ["au"] };
const C = { "ಕ": "k", "ಖ": "kh", "ಗ": "g", "ಘ": "gh", "ಙ": "ng", "ಚ": "c", "ಛ": "ch", "ಜ": "j", "ಝ": "jh", "ಞ": "ny",
  "ಟ": "tt", "ಠ": "tth", "ಡ": "dd", "ಢ": "ddh", "ಣ": "nn", "ತ": "t", "ಥ": "th", "ದ": "d", "ಧ": "dh", "ನ": "n",
  "ಪ": "p", "ಫ": "ph", "ಬ": "b", "ಭ": "bh", "ಮ": "m", "ಯ": "y", "ರ": "r", "ಱ": "r", "ಲ": "l", "ವ": "v",
  "ಶ": "sh", "ಷ": "ssh", "ಸ": "s", "ಹ": "h", "ಳ": "ll", "ೞ": "ll", "ೝ": "n" };
const NUKTA_MAP = { ph: "f", j: "z" };
const VIRAMA = "್", ANUSVARA = "ಂ", VISARGA = "ಃ", CANDRABINDU = "ಁ", NUKTA = "಼";
const IGNORED = new Set(["ಽ", "ೕ", "ೖ", "‌", "‍"]);
const PLACE = {};
for (const [nasal, group] of [["ng", "k kh g gh"], ["ny", "c ch j jh"], ["nn", "tt tth dd ddh"], ["n", "t th d dh"], ["m", "p ph b bh"]]) {
  for (const p of group.split(" ")) PLACE[p] = nasal;
}
const KANNADA_WORD = new RegExp(`^[${KN}\\u200c\\u200d]+$`, "u");
const HAS_KANNADA = new RegExp(`[${KN}]`, "u");

/** Kannada word -> phones without the kn_ prefix, or null if unsupported. */
export function kannadaWordPhones(word) {
  word = word.normalize("NFC");
  if (!word || !KANNADA_WORD.test(word) || /[೦-೯]/u.test(word)) return null;
  const out = [], pending = [], n = word.length;
  let i = 0;
  while (i < n) {
    const ch = word[i];
    if (IGNORED.has(ch)) i += 1;
    else if (ch in C) {
      let cons = C[ch], j = i + 1;
      if (j < n && word[j] === NUKTA) { cons = NUKTA_MAP[cons] ?? cons; j += 1; }
      while (j < n && IGNORED.has(word[j])) j += 1;
      out.push(cons);
      if (j < n && word[j] === VIRAMA) j += 1;
      else if (j < n && word[j] in M) { out.push(...M[word[j]]); j += 1; }
      else out.push("a");
      i = j;
    } else if (ch in V) { out.push(...V[ch]); i += 1; }
    else if (ch === ANUSVARA || ch === CANDRABINDU) { pending.push(out.length); out.push("m"); i += 1; }
    else if (ch === VISARGA) { out.push("h"); i += 1; }
    else if (ch in M || ch === VIRAMA || ch === NUKTA) i += 1;
    else return null;
  }
  for (const pos of pending) out[pos] = PLACE[out[pos + 1]] ?? "m";
  return out.length ? out : null;
}

const ROMAN = [["chh", ["ch"]], ["sh", ["sh"]], ["ch", ["c"]], ["th", ["t"]], ["dh", ["dh"]], ["bh", ["bh"]],
  ["kh", ["kh"]], ["gh", ["gh"]], ["ph", ["ph"]], ["aa", ["aa"]], ["ee", ["ii"]], ["ii", ["ii"]], ["oo", ["uu"]],
  ["uu", ["uu"]], ["ai", ["ai"]], ["au", ["au"]], ["ou", ["au"]], ["ow", ["au"]], ["ei", ["ee"]], ["a", ["a"]],
  ["e", ["e"]], ["i", ["i"]], ["o", ["o"]], ["u", ["u"]], ["b", ["b"]], ["c", ["k"]], ["d", ["dd"]], ["f", ["f"]],
  ["g", ["g"]], ["h", ["h"]], ["j", ["j"]], ["k", ["k"]], ["l", ["l"]], ["m", ["m"]], ["n", ["n"]], ["p", ["p"]],
  ["q", ["k"]], ["r", ["r"]], ["s", ["s"]], ["t", ["tt"]], ["v", ["v"]], ["w", ["v"]], ["x", ["k", "s"]],
  ["y", ["y"]], ["z", ["z"]]];
const KN_VOWEL_SET = new Set(KN_VOWELS);
const KN_TO_ARPABET = { a: "AH", aa: "AA", i: "IH", ii: "IY", u: "UH", uu: "UW", e: "EH", ee: "EY", o: "OW", oo: "OW",
  ai: "AY", au: "AW", k: "K", kh: "K", g: "G", gh: "G", ng: "NG", c: "CH", ch: "CH", j: "JH", jh: "JH", ny: "N",
  tt: "T", tth: "T", dd: "D", ddh: "D", nn: "N", t: "T", th: "T", d: "D", dh: "D", n: "N", p: "P", ph: "P",
  b: "B", bh: "B", m: "M", y: "Y", r: "R", l: "L", v: "V", ll: "L", sh: "SH", ssh: "SH", s: "S", h: "HH", f: "F", z: "Z" };

/** 'Bengaluru' -> ['b','e','n','g','a','l','u','r','u'] (Kannada phones, no prefix). */
export function romanizedWordPhones(word) {
  word = word.toLowerCase().replace(/[^a-z]/g, "");
  const out = [];
  let i = 0;
  while (i < word.length) {
    if (word[i] === "y" && (i + 1 === word.length || !"aeiou".includes(word[i + 1])) && out.length
        && !KN_VOWEL_SET.has(out[out.length - 1])) {
      out.push(i + 1 === word.length ? "i" : "ai");
      i += 1;
      continue;
    }
    for (const [letters, phones] of ROMAN) {
      if (word.startsWith(letters, i)) { out.push(...phones); i += letters.length; break; }
    }
  }
  return out;
}

export function romanizedArpabet(word) {
  const out = [];
  let stressed = false;
  for (const phone of romanizedWordPhones(word)) {
    let symbol = KN_TO_ARPABET[phone];
    if (KN_VOWEL_SET.has(phone)) { symbol += stressed ? "0" : "1"; stressed = true; }
    out.push(symbol);
  }
  return out;
}

// ------------------------------------------------------------------ bilingual frontend
const KN_PUNCT = { "।": ".", "॥": ".", "…": ".", "—": ",", "–": ",", "‘": "'", "’": "'",
  "“": " ", "”": " ", '"': " ", "(": " ", ")": " ", "[": " ", "]": " ",
  "{": " ", "}": " ", "/": " ", "-": " ", "_": " ", "*": " " };
const TOKEN = rx(String.raw`[${KN}‌‍]+|[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]|\S`);
const LATIN_WORD = /^[A-Za-z]+(?:'[A-Za-z]+)*$/u;
const DIGIT_CHAR = /^[\p{Nd}²³¹⁰-⁹₀-₉①-⑨⑴-⑼⒈-⒐⓵-⓽❶-❾➀-➈➊-➒]$/u;

export function normalize(text) {
  text = translate(text.normalize("NFC"), KN_PUNCT);
  text = text.replaceAll("‌", "").replaceAll("‍", "");
  text = text.replace(/([.?!])\1+/gu, "$1");
  return collapse(text);
}

/**
 * Text in Kannada, English, or both -> { normalized, symbols, ids, unknown, skipped }.
 * Same options as kannada.phonemize. `lenient: true` (used by the web app) never throws on user
 * text: characters it cannot read are skipped and listed in `skipped` instead.
 */
export function phonemize(text, { strict = true, overrides = null, guessUnknown = true, lenient = false } = {}) {
  const merged = { ...INDIAN_ENGLISH_LEXICON, ...(overrides || {}) };
  const skipped = [];
  if (lenient) strict = false;
  if (!HAS_KANNADA.test(text)) {
    const english = normalizeIndianEnglish(text);
    const normalized = normalizeEnglish(english, { lenient, skipped });
    if (guessUnknown) {
      const lowered = new Set(Object.keys(merged).map((k) => k.toLowerCase()));
      for (const [w] of normalized.matchAll(/[A-Za-z]+(?:'[A-Za-z]+)*/gu)) {
        const word = w.toLowerCase();
        if (lookup(word) === undefined && !lowered.has(word) && /^[a-z]+$/.test(word)) merged[word] = romanizedArpabet(word);
      }
    }
    const { tokens, unknown } = phonemizeEnglish(normalized, { strict, overrides: merged });
    return result(normalized, tokens, unknown, skipped);
  }
  const normalized = normalize(normalizeKannadaText(text));
  const over = new Map();
  for (let [word, phones] of Object.entries(merged)) {
    word = word.normalize("NFC");
    over.set(HAS_KANNADA.test(word) ? word : word.toLowerCase(), phones);
  }
  const tokens = ["<bos>"], unknown = [];
  for (const [item] of normalized.matchAll(TOKEN)) {
    if (SPECIAL.includes(item)) {
      if (tokens[tokens.length - 1] !== "<bos>") tokens.push(item);
      continue;
    }
    let phones;
    if (KANNADA_WORD.test(item)) {
      phones = over.get(item);
      if (phones === undefined) {
        const bare = kannadaWordPhones(item);
        if (bare === null) {
          if (lenient) { skipped.push(item); continue; }
          throw new FrontendError("kannada-word", `cannot read Kannada word ${item}; write numbers in words or add a pronunciation override`);
        }
        phones = bare.map((p) => "kn_" + p);
      }
    } else if (LATIN_WORD.test(item)) {
      const word = item.toLowerCase();
      phones = over.get(word);
      if (phones === undefined) phones = lookup(word);
      if (phones === undefined) {
        unknown.push(word);
        phones = guessUnknown ? romanizedWordPhones(word).map((p) => "kn_" + p) : spell(word);
      }
    } else if (lenient) {
      skipped.push(item);
      continue;
    } else if (DIGIT_CHAR.test(item)) {
      throw new FrontendError("number", `could not read the number ${item}`);
    } else {
      throw new FrontendError("unsupported", `unsupported character in Kannada text: ${item}`);
    }
    if (tokens[tokens.length - 1] !== "<bos>" && tokens[tokens.length - 1] !== "<space>") tokens.push("<space>");
    for (const p of phones) {
      if (!SYMBOL_TO_ID.has(p)) throw new FrontendError("phone", `unknown phone ${p} in ${item}`);
      tokens.push(p);
    }
  }
  if (unknown.length && strict && !guessUnknown) {
    throw new FrontendError("oov", "out-of-vocabulary words require pronunciation overrides: " + [...new Set(unknown)].sort().join(", "));
  }
  tokens.push("<eos>");
  return result(normalized, tokens, unknown, skipped);
}

function result(normalized, symbols, unknown, skipped) {
  return { normalized, symbols, ids: symbols.map((s) => SYMBOL_TO_ID.get(s)), unknown, skipped };
}

// ------------------------------------------------------------------ utterance splitting (web only)
const ID = Object.fromEntries(SPECIAL.map((s, i) => [s, i]));
const SENTENCE_END = new Set([ID["."], ID["?"], ID["!"]]);
const CLAUSE_END = new Set([ID[","], ID[";"], ID[":"]]);
const NUM_SPECIAL = SPECIAL.length;

/**
 * Split a token id list (<bos> ... <eos>) into utterances of at most `maxTokens` body tokens:
 * at sentence ends first, then at commas, then between words. Each returned list starts with
 * <bos> and ends with <eos>; pieces without any phone are dropped.
 */
export function splitUtterances(ids, maxTokens = 160) {
  const body = ids.slice(1, -1);
  const cutAfter = (list, test) => {
    const out = [];
    let cur = [];
    for (const id of list) {
      cur.push(id);
      if (test(id)) { out.push(cur); cur = []; }
    }
    if (cur.length) out.push(cur);
    return out;
  };
  const pack = (pieces) => {
    const out = [];
    let cur = [];
    for (const p of pieces) {
      if (cur.length && cur.length + p.length > maxTokens) { out.push(cur); cur = []; }
      cur = cur.concat(p);
    }
    if (cur.length) out.push(cur);
    return out;
  };
  const chunks = [];
  for (const sentence of cutAfter(body, (id) => SENTENCE_END.has(id))) {
    if (sentence.length <= maxTokens) { chunks.push(sentence); continue; }
    for (const clause of pack(cutAfter(sentence, (id) => CLAUSE_END.has(id)))) {
      if (clause.length <= maxTokens) { chunks.push(clause); continue; }
      // between words: cut before each <space>, never inside a word
      const words = [];
      let cur = [];
      for (const id of clause) {
        if (id === ID["<space>"] && cur.length) { words.push(cur); cur = []; }
        cur.push(id);
      }
      if (cur.length) words.push(cur);
      chunks.push(...pack(words));
    }
  }
  return chunks
    .map((c) => {
      let a = 0, b = c.length;
      while (a < b && c[a] === ID["<space>"]) a++;
      while (b > a && c[b - 1] === ID["<space>"]) b--;
      return [ID["<bos>"], ...c.slice(a, b), ID["<eos>"]];
    })
    .filter((c) => c.some((id) => id >= NUM_SPECIAL));
}

/** True when the token list contains something to say (the runtime returns no audio otherwise). */
export const hasSpeech = (ids) => ids.some((id) => id >= NUM_SPECIAL);
