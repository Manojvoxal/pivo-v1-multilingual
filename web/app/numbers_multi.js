// Read numbers, dates, times, money, units and symbols aloud in 10 Indian languages
// (hi mr gu pa bn or as ta te ml). Line-by-line port of
// training/data_pipeline_multi/pivo_multi/numnorm.py; both read the same word data
// (numwords.js is numwords.json as a module). web/tests/numbers_parity.* checks that the two
// give identical text. Kannada and English keep their own normalisers (frontend.js). semiotic.js (phones,
// e-mail / URLs, date formats, fractions, math, serial codes ...) runs first, as in Python.

import DATA from "./numwords.js";
import { prenormalize, WORDS as SEMIOTIC } from "./semiotic.js";

export const NUMBER_LANGS = Object.keys(DATA);
for (const [lang, D] of Object.entries(DATA)) D.measures = { ...SEMIOTIC[lang].units, ...D.measures };
// English words before a year in mixed text: "in 1947", "since 1857"
const EN_YEAR_BEFORE = ["in", "In", "since", "Since", "from", "From", "till", "Till", "until", "Until", "year", "Year",
  "during", "During", "circa", "around", "by", "to", "before", "after", "Before", "After"];
const YEAR_AFTER_EXTRA = ["AD", "BC", "CE", "BCE"];
/** First word / suffix after the year in the language's decade phrase ("का" in "{y} का दशक"). */
function decadeFollow(lang) {
  const rest = SEMIOTIC[lang].decade.split("{y}").slice(1).join("{y}").replace(/^-+/u, "");
  return rest.startsWith(" ") ? rest.split(" ")[1] : rest.split(" ")[0];
}
const SCRIPT_BASE = { hi: 0x900, mr: 0x900, bn: 0x980, as: 0x980, pa: 0xA00, gu: 0xA80, or: 0xB00, ta: 0xB80, te: 0xC00, ml: 0xD00 };
const MAX_DIGITS = 15;
const ACRONYM_KEEP = new Set(["NASA", "ISRO", "COVID", "UNESCO", "UNICEF", "NATO", "AIDS", "SAARC", "ASEAN", "OPEC",
  "NABARD", "SEBI", "NEET", "OK"]);
const EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];
const EN_SCALES = { thousand: 3, k: 3, K: 3, lakh: 5, lakhs: 5, lac: 5, lacs: 5, crore: 7, crores: 7, cr: 7, Cr: 7,
  million: 6, mn: 6, billion: 9, bn: 9 };
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

const CHAR_MAP = {};
for (let base = 0x900; base < 0xD80; base += 0x80) for (let d = 0; d < 10; d++) CHAR_MAP[String.fromCodePoint(base + 0x66 + d)] = String(d);
for (const c of [0xA0, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F,
  0x205F, 0x3000, 0x09, 0x0A, 0x0D]) CHAR_MAP[String.fromCodePoint(c)] = " ";
CHAR_MAP["−"] = "-";

// ------------------------------------------------------------------ Brahmic letters
const cp = (ch) => ch.codePointAt(0);
const off = (ch) => cp(ch) & 0x7F;
const brahmic = (ch) => cp(ch) >= 0x900 && cp(ch) < 0xD80;
const isVowel = (ch) => brahmic(ch) && off(ch) >= 0x05 && off(ch) <= 0x14;
const isSign = (ch) => brahmic(ch) && off(ch) >= 0x3E && off(ch) <= 0x4C;
const isVirama = (ch) => brahmic(ch) && off(ch) === 0x4D;
function isConsonant(ch) {
  if (!brahmic(ch)) return false;
  const o = off(ch);
  return (o >= 0x15 && o <= 0x39) || (o >= 0x58 && o <= 0x5F) || o === 0x3C;
}
const vowelSign = (ch) => (off(ch) === 0x05 ? "" : String.fromCodePoint(cp(ch) - 0x06 + 0x3E));
const signVowel = (ch) => String.fromCodePoint(cp(ch) - 0x3E + 0x06);
const endsWith = (s, end) => s.length >= end.length && s.slice(s.length - end.length) === end;
const cutEnd = (s, end) => s.slice(0, s.length - end.length);

/** Glue a case / postposition suffix onto the last word, with the language's sandhi. */
export function attach(words, suffix, lang) {
  if (!suffix) return words;
  const D = DATA[lang];
  const cut = words.lastIndexOf(" ");
  const head = words.slice(0, cut + 1);
  let last = words.slice(cut + 1);
  let s = suffix;
  if (isSign(s[0])) s = signVowel(s[0]) + s.slice(1);
  if (D.suffix_map && has(D.suffix_map, s)) s = D.suffix_map[s];
  const vowel = isVowel(s[0]);
  for (const [end, cls, rep] of D.sandhi || []) {
    if (endsWith(last, end) && ((cls === "V" && vowel) || (cls === "C" && !vowel) || (cls !== "V" && cls !== "C" && cls.includes(s[0])))) {
      last = cutEnd(last, end) + rep;
      break;
    }
  }
  if (vowel && D.sandhi) {
    const t = last[last.length - 1];
    const sign = vowelSign(s[0]);
    if (isVirama(t)) last = last.slice(0, -1) + sign;
    else if (isConsonant(t)) last = last + sign;
    else last = last + D.glide + sign;
    return head + last + s.slice(1);
  }
  return head + last + s;
}

// ------------------------------------------------------------------ numbers
function attr(D, words) {
  const table = D.attr;
  if (!table) return words;
  const cut = words.lastIndexOf(" ");
  const last = words.slice(cut + 1);
  return has(table, last) ? words.slice(0, cut + 1) + table[last] : words;
}
const pick = (forms, pos) => (pos < forms.length ? forms[pos] : forms[0]);

function scale(D, lang, name, k, pos) {
  const S = D.scales[name];
  const over = S.over && has(S.over, String(k)) ? S.over[String(k)] : null;
  if (over) return pick(over, pos);
  const form = pick(S.forms, pos);
  if (S.join === "space") return cardinal(k, lang, true) + " " + form;
  let mult = cardinal(k, lang);
  for (const [end, rep] of S.pre || []) {
    if (endsWith(mult, end)) { mult = cutEnd(mult, end) + rep; break; }
  }
  return attach(mult, form, lang);
}

/** n >= 0 in words, Indian grouping. attr: the form used before a noun. */
export function cardinal(n, lang, isAttr = false) {
  const D = DATA[lang];
  if (n < 100) return isAttr ? attr(D, D.units[n]) : D.units[n];
  const crore = (n - (n % 1e7)) / 1e7; let rest = n % 1e7;
  const lakh = (rest - (rest % 1e5)) / 1e5; rest %= 1e5;
  const thousand = (rest - (rest % 1000)) / 1000; rest %= 1000;
  const hundred = (rest - (rest % 100)) / 100, tail = rest % 100;
  const segs = [["crore", crore], ["lakh", lakh], ["thousand", thousand], ["hundred", hundred]];
  const words = [];
  segs.forEach(([name, k], i) => {
    if (!k) return;
    const final = tail === 0 && segs.slice(i + 1).every(([, v]) => v === 0);
    const pos = final ? (isAttr ? 2 : 0) : 1;
    words.push(name === "hundred" ? pick(D.hundreds[k - 1], pos) : scale(D, lang, name, k, pos));
  });
  if (tail) words.push(isAttr ? attr(D, D.units[tail]) : D.units[tail]);
  return words.join(" ");
}

export const digits = (s, lang) => Array.from(s, (c) => DATA[lang].units[Number(c)]).join(" ");

function integerWords(s, lang, isAttr = false) {
  s = s.replace(/^0+/, "") || "0";
  return s.length > MAX_DIGITS ? digits(s, lang) : cardinal(Number(s), lang, isAttr);
}

function partition(s, sep) {
  const i = s.indexOf(sep);
  return i < 0 ? [s, ""] : [s.slice(0, i), s.slice(i + sep.length)];
}

export function numberText(raw, lang, isAttr = false) {
  const D = DATA[lang];
  const [whole, frac] = partition(raw.replaceAll(",", ""), ".");
  let words = whole ? integerWords(whole, lang, isAttr && !frac) : D.units[0];
  if (frac) words += " " + D.point + " " + digits(frac, lang);
  return words;
}

export function yearWords(y, lang) {
  const D = DATA[lang], Y = D.year;
  if (!Y || !(y >= 1100 && y <= 1999)) return cardinal(y, lang);
  const c = Math.floor(y / 100), r = y % 100;
  const word = r === 0 && Y.length > 2 ? Y[2] : Y[0];
  const head = D.units[c] + (Y[1] ? "" : " ") + word;
  return r === 0 ? head : head + " " + D.units[r];
}

export function ordinal(n, lang, ending = null) {
  const D = DATA[lang], O = D.ordinal;
  const e = O.endings[ending || O.default];
  const stem = has(O.stems, String(n)) ? O.stems[String(n)] : null;
  if (O.mode === "concat") return stem !== null ? stem + e[0] : integerWords(String(n), lang) + e[1];
  if (stem !== null) return attach(stem, e[0], lang);
  return attach(integerWords(String(n), lang), e[1], lang);
}

const plural = (D, key, one) => (one ? D.currency[key][0] : D.currency[key][1]);
function measure(D, unit, one) {
  const w = D.measures[unit];
  return typeof w === "string" ? w : one ? w[0] : w[1];
}
function shift(amount, exp) {
  const [whole, frac] = partition(amount.replaceAll(",", ""), ".");
  if (frac.length > exp) return null;
  return ((whole || "0") + frac.padEnd(exp, "0")).replace(/^0+/, "") || "0";
}

// ------------------------------------------------------------------ patterns
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const cmp = (a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0);
const alt = (words) => [...words].sort(cmp).map(escapeRe).join("|");
const R = (src) => new RegExp(src, "gu");
const hex = (n) => "\\u" + n.toString(16).toUpperCase().padStart(4, "0");

const PATTERNS = {};
function patterns(lang) {
  if (PATTERNS[lang]) return PATTERNS[lang];
  const D = DATA[lang];
  const b = SCRIPT_BASE[lang];
  const L = `${hex(b)}-${hex(b + 0x63)}${hex(b + 0x70)}-${hex(b + 0x7F)}\\u200C\\u200D`;
  const NUM = String.raw`[0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?`;
  const SUF = `(?:-?([${L}]+))?`;
  const units = alt(Object.keys(D.measures));
  const nativeScales = new Set();
  for (const S of Object.values(D.scales)) for (const f of S.forms) nativeScales.add(f);
  const scaleAlt = alt([...Object.keys(EN_SCALES), ...[...nativeScales].sort()]);
  const cur = { inr: ["₹", "Rs.", "Rs", "INR", "रु.", ...D.currency_abbr], usd: ["$", "US$", "USD"], eur: ["€", "EUR"], gbp: ["£", "GBP"] };
  const currencyOf = {};
  for (const [k, v] of Object.entries(cur)) for (const s of v) currencyOf[s] = k;
  const months = [...D.months, ...(D.month_alt || []), ...EN_MONTHS];
  const abbr = Object.entries(D.abbreviations).sort((a, b) => cmp(a[0], b[0])).map(([k, v]) => {
    const right = k === "No." ? "(?= ?[0-9])" : k[k.length - 1] !== "." ? `(?![A-Za-z${L}])` : "";
    return [R(`(?<![A-Za-z${L}])${escapeRe(k)}${right}`), " " + v + " "];
  });
  const clockWords = !D.clock_words.length ? "()"
    : D.clock_suffix ? `(?: (?:${alt(D.clock_words)})([${L}]*))?`
      : `(?: (?:${alt(D.clock_words)})()(?![${L}]))?`;
  const afterNumber = D.abbr_after_number || {};
  const P = {
    L,
    abbr,
    abbrAfterNumber: Object.keys(afterNumber).length ? R(`(?<=[0-9])( ?)(${alt(Object.keys(afterNumber))})`) : null,
    initials: R(`(?<![${L}])([${L}]+)\\.(?= ?[${L}])`),
    mobile: R(`(\\+91[ -]?)?(?<![0-9])([6-9][0-9]{4})[ -]?([0-9]{5})(?![0-9])${SUF}`),
    landline: R(`(?<![0-9])(0[0-9]{2,4})[ -]([0-9]{6,8})(?![0-9])${SUF}`),
    pin: R(`(${alt([...D.pin_words, "PIN", "Pin", "pin", "Pincode", "PIN code"])}|[${L}]+ ?-) ?:? ?([0-9]{6})(?![0-9])${SUF}`),
    iso: R(`(?<![0-9])([0-9]{4})-([0-9]{2})-([0-9]{2})(?![0-9])${SUF}`),
    dmy: R(`(?<![0-9.])([0-9]{1,2})[/.-]([0-9]{1,2})[/.-]([0-9]{4})(?![0-9])${SUF}`),
    monthYear: R(`(?<![A-Za-z${L}])(${alt(months)}),? ([0-9]{4})(?![0-9])${SUF}`),
    time: R(`(?<![0-9])([01]?[0-9]|2[0-3]):([0-5][0-9])(?::[0-5][0-9])?(?: ?([AaPp](?:\\.[Mm]\\.|[Mm])))?(?![A-Za-z])${clockWords}${SUF}`),
    timeAmpm: R(`(?<![0-9.,])(1[0-2]|0?[1-9]) ?([AaPp](?:\\.[Mm]\\.|[Mm]))(?![A-Za-z])${SUF}`),
    money: R(`(${alt(Object.values(cur).flat())}) ?(${NUM})(?: ?(${scaleAlt})(?![A-Za-z${L}]))?(?![0-9])${SUF}`),
    currencyOf,
    percent: R(`(${NUM}) ?%${SUF}`),
    range: R(`(?<![0-9.,])([0-9]+) ?[-–] ?([0-9]+)(?![0-9]|[.,][0-9])(?: ?(${units})(?![A-Za-z])| ?(%))?${SUF}`),
    negative: R(`(?<![0-9A-Za-z${L}])-(?=[0-9])`),
    units: R(`(${NUM}) ?(${units})(?![A-Za-z])`),
    enOrdinal: R("(?<![0-9])([0-9]+)(?:st|nd|rd|th)(?![A-Za-z])"),
    long: R(`(?<![.,0-9])(0[0-9]+|[0-9]{10,})(?![0-9])${SUF}`),
    number: R(`(${NUM})(?![0-9])${SUF}`),
    acronym: R("(?<![A-Za-z])[A-Z]{2,6}(?![A-Za-z])"),
    spacePunct: R(" +([.,?!;:।])"),
    spaces: R(" {2,}"),
    yearBefore: D.year_before.length ? new RegExp(`(?:^|[^${L}])(?:${alt(D.year_before)}) $`, "u") : null,
    yearBeforeEn: new RegExp(`(?:^|[^A-Za-z${L}])(?:${alt(EN_YEAR_BEFORE)}) $`, "u"),
    yearAfter: [...D.year_after, ...YEAR_AFTER_EXTRA, decadeFollow(lang)],
    nextLetter: new RegExp(`^ [${L}]`, "u"),
    months,
  };
  PATTERNS[lang] = P;
  return P;
}

/** re.sub with a function receiving (m, offset, input); m[k] is "" -> undefined like Python's None. */
function sub(re, text, fn) {
  let out = "", last = 0;
  for (const m of text.matchAll(re)) {
    out += text.slice(last, m.index) + fn(m, m.index, text);
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

// ------------------------------------------------------------------ rules
function period(D, hour, ampm) {
  const P = D.periods;
  if (ampm === "am") return hour === 12 ? P.midnight : hour <= 3 ? P.night : P.morning;
  if (hour === 12 || hour <= 3) return P.noon;
  return hour <= 6 ? P.evening : P.night;
}

function clock(D, lang, hour, minute, ampm) {
  if (hour > 12) { ampm = "pm"; hour -= 12; } // 24-hour clock (an am/pm after it is ignored)
  else if (hour === 0) { ampm = "am"; hour = 12; }
  const p = ampm ? period(D, hour, ampm) : "";
  const C = D.clock;
  const m = cardinal(minute, lang, true);
  let tpl, h;
  if (hour === 1 && C.one) { tpl = C.one[minute ? 1 : 0]; h = ""; }
  else {
    tpl = minute ? C.hm : C.h;
    h = C.hours ? C.hours[hour - 1] : cardinal(hour, lang, true);
  }
  return tpl.replaceAll("{p}", p).replaceAll("{h}", h).replaceAll("{m}", m).split(" ").filter(Boolean).join(" ");
}

function date(D, lang, day, month, year, suffix) {
  if (!(month >= 1 && month <= 12 && day >= 1 && day <= 31)) return null;
  const words = `${cardinal(day, lang)} ${D.months[month - 1]} ${yearWords(year, lang)}`;
  return " " + attach(words, suffix, lang) + " ";
}

function suffixWords(raw, suffix, lang, isAttr, year) {
  const D = DATA[lang];
  const plain = !raw.includes(",") && !raw.includes(".");
  const n = plain && raw.length <= MAX_DIGITS ? Number(raw) : null;
  if (suffix && n !== null) {
    const day = D.day_suffixes && has(D.day_suffixes, suffix) ? D.day_suffixes[suffix] : null;
    if (day && day[0].includes(n)) {
      const form = day[1];
      if (typeof form === "object") return form[String(n)];
      if (form === "ই") return cardinal(n, lang) + form;
      if (day[0].length === 1) return form;
      return attach(cardinal(n, lang), form, lang);
    }
    const O = D.ordinal.suffixes;
    const s = isSign(suffix[0]) ? signVowel(suffix[0]) + suffix.slice(1) : suffix;
    const hit = has(O, suffix) ? O[suffix] : has(O, s) ? O[s] : null;
    if (hit && (hit.length === 1 || hit[1].includes(n))) return ordinal(n, lang, hit[0]);
  }
  let words;
  if (year && n !== null) words = yearWords(n, lang);
  else words = numberText(raw, lang, isAttr || Boolean(suffix && D.suffix_attr));
  return attach(words, suffix, lang);
}

export function normalizeNumbers(text, lang) {
  if (!has(DATA, lang)) throw new Error(`no number data for ${lang}`);
  const D = DATA[lang];
  const P = patterns(lang);
  text = prenormalize(text, lang, { digits: D.units.slice(0, 10), months: D.months, point: D.point, plus: D.plus,
    range: D.range });
  text = Array.from(text.normalize("NFC"), (c) => (has(CHAR_MAP, c) ? CHAR_MAP[c] : c)).join("");
  for (const [re, spoken] of P.abbr) text = text.replace(re, () => spoken);
  if (P.abbrAfterNumber) text = sub(P.abbrAfterNumber, text, (m) => " " + D.abbr_after_number[m[2]] + " ");
  const letters = new Set(D.letters);
  text = sub(P.initials, text, (m) => (letters.has(m[1]) ? m[1] + " " : m[0]));

  text = sub(P.mobile, text, (m) => {
    const plus = m[1] ? D.plus + " " + digits("91", lang) + " " : "";
    return " " + plus + attach(digits(m[2] + m[3], lang), m[4], lang) + " ";
  });
  text = sub(P.landline, text, (m) => " " + attach(digits(m[1] + m[2], lang), m[3], lang) + " ");
  text = sub(P.pin, text, (m) => " " + m[1].replace(/[- ]+$/, "") + " " + attach(digits(m[2], lang), m[3], lang) + " ");
  text = sub(P.iso, text, (m) => date(D, lang, Number(m[3]), Number(m[2]), Number(m[1]), m[4]) ?? m[0]);
  text = sub(P.dmy, text, (m) => date(D, lang, Number(m[1]), Number(m[2]), Number(m[3]), m[4]) ?? m[0]);
  text = sub(P.monthYear, text, (m) => ` ${m[1]} ` + attach(yearWords(Number(m[2]), lang), m[3], lang) + " ");
  text = sub(P.time, text, (m) => {
    const ampm = (m[3] || "").replaceAll(".", "").toLowerCase() || null;
    return " " + attach(clock(D, lang, Number(m[1]), Number(m[2]), ampm), m[5] || m[4], lang) + " ";
  });
  text = sub(P.timeAmpm, text, (m) => " " + attach(clock(D, lang, Number(m[1]), 0, m[2].replaceAll(".", "").toLowerCase()), m[3], lang) + " ");

  text = sub(P.money, text, (m) => {
    const kind = P.currencyOf[m[1]];
    const amount = m[2], sc = m[3], suffix = m[4];
    const small = kind !== "inr" ? "cent" : "paise";
    if (sc) {
      let exp = has(EN_SCALES, sc) ? EN_SCALES[sc] : undefined;
      if (exp === undefined) exp = [["thousand", 3], ["lakh", 5], ["crore", 7]].find(([name]) => D.scales[name].forms.includes(sc))[1];
      const whole = shift(amount, exp);
      let words;
      if (whole !== null) words = integerWords(whole, lang, true) + " " + plural(D, kind, whole === "1");
      else {
        const name = { 3: "thousand", 5: "lakh", 7: "crore" }[exp];
        const scaleWord = name ? pick(D.scales[name].forms, 2) : sc;
        words = numberText(amount, lang) + " " + scaleWord + " " + plural(D, kind, false);
      }
      return " " + attach(words, suffix, lang) + " ";
    }
    let [whole, frac] = partition(amount.replaceAll(",", ""), ".");
    whole = whole || "0";
    let words = integerWords(whole, lang, true) + " " + plural(D, kind, whole.replace(/^0+/, "") === "1");
    const subUnits = frac ? Number(frac.padEnd(2, "0").slice(0, 2)) : 0;
    if (subUnits) {
      const sep = D.money_and ? " " + D.money_and + " " : " ";
      words += sep + cardinal(subUnits, lang, true) + " " + plural(D, small, subUnits === 1);
    }
    return " " + attach(words, suffix, lang) + " ";
  });
  text = sub(P.range, text, (m) => {
    let a = attach(integerWords(m[1], lang), D.range[0], lang);
    if (D.range[1]) a += " " + D.range[1];
    let b = integerWords(m[2], lang, Boolean(m[3] || m[4]));
    if (m[3]) b += " " + measure(D, m[3], false);
    else if (m[4]) b += " " + D.percent;
    return " " + a + " " + attach(b, m[5], lang) + " ";
  });
  text = sub(P.percent, text, (m) => " " + attach(numberText(m[1], lang, true) + " " + D.percent, m[2], lang) + " ");
  text = text.replace(P.negative, " " + D.minus + " ");
  text = sub(P.units, text, (m) => {
    const one = m[1].replaceAll(",", "") === "1";
    return " " + numberText(m[1], lang, true) + " " + measure(D, m[2], one) + " ";
  });
  text = sub(P.enOrdinal, text, (m) => (m[1].length <= MAX_DIGITS ? " " + ordinal(Number(m[1]), lang) + " " : m[0]));
  text = sub(P.long, text, (m) => {
    const s = m[1];
    if (s[0] !== "0" && s.length <= MAX_DIGITS && /^[0-9]+?0{5,}$/.test(s)) return " " + attach(cardinal(Number(s), lang), m[2], lang) + " ";
    return " " + attach(digits(s, lang), m[2], lang) + " ";
  });
  text = sub(P.number, text, (m, index, input) => {
    const raw = m[1];
    const suffix = (m[2] || "").replaceAll("‌", "").replaceAll("‍", "");
    const before = input.slice(0, index), after = input.slice(index + m[0].length);
    let year = false;
    if (/^1[1-9][0-9]{2}$/.test(raw)) {
      const nxt = after.startsWith(" ") ? after.slice(1).split(" ")[0] : "";
      year = Boolean(P.yearBefore && P.yearBefore.test(before)) || P.yearBeforeEn.test(before)
        || P.yearAfter.some((w) => nxt.startsWith(w))
        || P.yearAfter.some((w) => suffix.startsWith(w)) || P.months.includes(nxt);
    }
    const isAttr = !suffix && P.nextLetter.test(after);
    return " " + suffixWords(raw, suffix, lang, isAttr, year) + " ";
  });
  text = sub(P.acronym, text, (m) => {
    const w = m[0];
    if (ACRONYM_KEEP.has(w)) return w.toLowerCase();
    return " " + Array.from(w, (c) => D.letters[c.charCodeAt(0) - 65]).join(" ") + " ";
  });
  for (const [sym, word] of Object.entries(D.symbols)) text = text.replaceAll(sym, " " + word + " ");
  text = text.replaceAll("%", " " + D.percent + " ").replaceAll("°", " " + measure(D, "°", false) + " ");
  text = text.replace(P.spacePunct, "$1");
  return text.replace(P.spaces, " ").replace(/^ +| +$/g, "");
}
