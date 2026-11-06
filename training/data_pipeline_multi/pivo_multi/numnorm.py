"""Read numbers, dates, times, money, units and symbols aloud in 10 Indian languages.

    normalize_numbers(text, lang) -> text with digits and symbols written out as words
    lang: hi mr gu pa bn or as ta te ml   (Kannada and English keep their own normalisers)

Formal / news-reading style, the same rules as PiVo's Kannada normaliser plus a few more:

    phones, PIN codes      9876543210, +91 98765 43210, 080-2345 6789, PIN 560001  -> digit by digit
    dates                  15/08/1947, 2026-11-04, 15 अगस्त 1947, अगस्त 1947      -> day month year
    years                  1947 in a date or next to a year word (सन्, সালে, ମସିହା,
                           సంవత్సరం ...) -> "nineteen hundred forty-seven" style
    times                  10:30, 7:05 pm, 18:45                                 -> period + hour + minutes
    money                  ₹1,25,000.50  Rs 5 lakh  $3.20  €5  £2  (+ native abbreviations)
    percent, ranges        12.5%, 10-20, 10-15 km, 5-10%
    units                  km kg cm mm ml g l m °C ° GB MB hr min km/h mg sec
    ordinals               1st 2nd, 5वाँ 21वीं, ૫મો, ৫ম ২১তম, ১লা ২৫শে (Bengali dates),
                           5ஆம் 5வது, 5వ, 5-ാം 5ാമത്തെ
    negative numbers       -5, −12.5
    attached suffixes      2026లో, 2026இல், 2026-ൽ, ২০২৬-এ, ୨୦୨୬ରେ, 2026માં (with sandhi)
    long numbers           7+ digits or leading 0 -> digit by digit (round lakhs/crores read as numbers)
    numbers                lakh / crore grouping, decimals, native digits (०-९ ০-৯ ੦-੯ ...)
    acronyms, symbols      BJP -> letter names in the language's script; & + = @ % ° as words

All words live in numwords.json (shared with the web app's JavaScript port, which must give
exactly the same output; see web/tests/numbers_parity.*).
"""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA = json.loads(Path(__file__).with_name("numwords.json").read_text(encoding="utf-8"))
LANGS = tuple(DATA)
SCRIPT_BASE = {"hi": 0x900, "mr": 0x900, "bn": 0x980, "as": 0x980, "pa": 0xA00, "gu": 0xA80,
               "or": 0xB00, "ta": 0xB80, "te": 0xC00, "ml": 0xD00}
MAX_DIGITS = 15                     # longer integers are read digit by digit (JS Number is exact to 2^53)
ACRONYM_KEEP = {"NASA", "ISRO", "COVID", "UNESCO", "UNICEF", "NATO", "AIDS", "SAARC", "ASEAN", "OPEC",
                "NABARD", "SEBI", "NEET", "OK"}
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
EN_SCALES = {"thousand": 3, "k": 3, "K": 3, "lakh": 5, "lakhs": 5, "lac": 5, "lacs": 5, "crore": 7,
             "crores": 7, "cr": 7, "Cr": 7, "million": 6, "mn": 6, "billion": 9, "bn": 9}

_DIGITS = {}
for _base in range(0x900, 0xD80, 0x80):
    for _d in range(10):
        _DIGITS[_base + 0x66 + _d] = str(_d)
_SPACES = {c: " " for c in (0xA0, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
                            0x2009, 0x200A, 0x202F, 0x205F, 0x3000, 0x09, 0x0A, 0x0D)}
_SPACES[0x2212] = "-"               # minus sign


# ------------------------------------------------------------------ Brahmic letters
def _off(ch):
    return ord(ch) & 0x7F


def _is_vowel(ch):
    return 0x900 <= ord(ch) < 0xD80 and 0x05 <= _off(ch) <= 0x14


def _is_sign(ch):
    return 0x900 <= ord(ch) < 0xD80 and 0x3E <= _off(ch) <= 0x4C


def _is_virama(ch):
    return 0x900 <= ord(ch) < 0xD80 and _off(ch) == 0x4D


def _is_consonant(ch):
    if not 0x900 <= ord(ch) < 0xD80:
        return False
    o = _off(ch)
    return 0x15 <= o <= 0x39 or 0x58 <= o <= 0x5F or o == 0x3C


def _vowel_sign(ch):
    o = _off(ch)
    return "" if o == 0x05 else chr(ord(ch) - o + o - 0x06 + 0x3E)


def _sign_vowel(ch):
    o = _off(ch)
    return chr(ord(ch) - o + o - 0x3E + 0x06)


def attach(words, suffix, lang):
    """Glue a case / postposition suffix onto the last word, with the language's sandhi."""
    if not suffix:
        return words
    D = DATA[lang]
    cut = words.rfind(" ")
    head, last = words[:cut + 1], words[cut + 1:]
    s = suffix
    if _is_sign(s[0]):                           # "-ാം" written with a vowel sign
        s = _sign_vowel(s[0]) + s[1:]
    s = D.get("suffix_map", {}).get(s, s)
    vowel = _is_vowel(s[0])
    for end, cls, rep in D.get("sandhi", []):
        if last.endswith(end) and (cls == "V" and vowel or cls == "C" and not vowel
                                   or cls not in ("V", "C") and s[0] in cls):
            last = last[:len(last) - len(end)] + rep
            break
    if vowel and "sandhi" in D:
        t = last[-1]
        sign = _vowel_sign(s[0])
        if _is_virama(t):
            last = last[:-1] + sign
        elif _is_consonant(t):
            last = last + sign
        else:
            last = last + D["glide"] + sign
        return head + last + s[1:]
    return head + last + s


# ------------------------------------------------------------------ numbers
def _attr(D, words):
    table = D.get("attr")
    if not table:
        return words
    cut = words.rfind(" ")
    last = words[cut + 1:]
    return words[:cut + 1] + table.get(last, last) if last in table else words


def _pick(forms, pos):
    return forms[pos] if pos < len(forms) else forms[0]


def _scale(D, lang, name, k, pos):
    S = D["scales"][name]
    over = S.get("over", {}).get(str(k))
    if over:
        return _pick(over, pos)
    form = _pick(S["forms"], pos)
    if S["join"] == "space":
        return cardinal(k, lang, True) + " " + form
    mult = cardinal(k, lang)
    for end, rep in S.get("pre", []):
        if mult.endswith(end):
            mult = mult[:len(mult) - len(end)] + rep
            break
    return attach(mult, form, lang)


def cardinal(n, lang, attr=False):
    """n >= 0 in words, Indian grouping (crore, lakh, thousand, hundred). attr: the form used
    before a noun (Tamil ஒரு, Telugu ఒక / oblique వేల, Malayalam ഒരു)."""
    D = DATA[lang]
    if n < 100:
        return _attr(D, D["units"][n]) if attr else D["units"][n]
    crore, rest = divmod(n, 10 ** 7)
    lakh, rest = divmod(rest, 10 ** 5)
    thousand, rest = divmod(rest, 1000)
    hundred, tail = divmod(rest, 100)
    segs = [("crore", crore), ("lakh", lakh), ("thousand", thousand), ("hundred", hundred)]
    words = []
    for i, (name, k) in enumerate(segs):
        if not k:
            continue
        final = tail == 0 and all(v == 0 for _, v in segs[i + 1:])
        pos = (2 if attr else 0) if final else 1
        words.append(_pick(D["hundreds"][k - 1], pos) if name == "hundred" else _scale(D, lang, name, k, pos))
    if tail:
        words.append(_attr(D, D["units"][tail]) if attr else D["units"][tail])
    return " ".join(words)


def digits(s, lang):
    units = DATA[lang]["units"]
    return " ".join(units[int(c)] for c in s)


def integer_words(s, lang, attr=False):
    s = s.lstrip("0") or "0"
    return digits(s, lang) if len(s) > MAX_DIGITS else cardinal(int(s), lang, attr)


def number_text(raw, lang, attr=False):
    """'1,25,000.50' -> words (decimals read digit by digit after the point word)."""
    D = DATA[lang]
    raw = raw.replace(",", "")
    whole, _, frac = raw.partition(".")
    words = integer_words(whole, lang, attr and not frac) if whole else D["units"][0]
    if frac:
        words += " " + D["point"] + " " + digits(frac, lang)
    return words


def year_words(y, lang):
    D = DATA[lang]
    Y = D.get("year")
    if not Y or not 1100 <= y <= 1999:
        return cardinal(y, lang)
    c, r = divmod(y, 100)
    word = Y[2] if r == 0 and len(Y) > 2 else Y[0]
    head = D["units"][c] + ("" if Y[1] else " ") + word
    return head if r == 0 else head + " " + D["units"][r]


def ordinal(n, lang, ending=None):
    D = DATA[lang]
    O = D["ordinal"]
    e = O["endings"][ending or O["default"]]
    stem = O["stems"].get(str(n))
    if O["mode"] == "concat":
        return stem + e[0] if stem is not None else integer_words(str(n), lang) + e[1]
    if stem is not None:
        return attach(stem, e[0], lang)
    return attach(integer_words(str(n), lang), e[1], lang)


def _plural(D, key, one):
    forms = D["currency"][key]
    return forms[0] if one else forms[1]


def _measure(D, unit, one):
    w = D["measures"][unit]
    return w if isinstance(w, str) else (w[0] if one else w[1])


def _shift(amount, exp):
    """'2.5', 7 -> '25000000' (amount x 10^exp as an integer string), or None if not whole."""
    whole, _, frac = amount.replace(",", "").partition(".")
    if len(frac) > exp:
        return None
    return ((whole or "0") + frac.ljust(exp, "0")).lstrip("0") or "0"


# ------------------------------------------------------------------ patterns
def _esc(words):
    return "|".join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w)))


@lru_cache(maxsize=None)
def _patterns(lang):
    D = DATA[lang]
    b = SCRIPT_BASE[lang]
    L = f"{chr(b)}-{chr(b + 0x63)}{chr(b + 0x70)}-{chr(b + 0x7F)}‌‍"
    NUM = r"[0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?"
    SUF = rf"(?:-?([{L}]+))?"
    units = _esc(D["measures"])
    native_scales = set()
    for S in D["scales"].values():
        native_scales.update(S["forms"])
    scale_alt = _esc(list(EN_SCALES) + sorted(native_scales))
    cur = {"inr": ["₹", "Rs.", "Rs", "INR", "रु."] + D["currency_abbr"], "usd": ["$", "US$", "USD"],
           "eur": ["€", "EUR"], "gbp": ["£", "GBP"]}
    months = D["months"] + D.get("month_alt", []) + EN_MONTHS
    abbr = []
    for k, v in sorted(D["abbreviations"].items(), key=lambda kv: (-len(kv[0]), kv[0])):
        right = r"(?= ?[0-9])" if k == "No." else (rf"(?![A-Za-z{L}])" if k[-1] != "." else "")
        abbr.append((re.compile(rf"(?<![A-Za-z{L}]){re.escape(k)}{right}"), " " + v + " "))
    if not D["clock_words"]:
        clock_words = "()"
    elif D.get("clock_suffix"):
        clock_words = rf"(?: (?:{_esc(D['clock_words'])})([{L}]*))?"
    else:
        clock_words = rf"(?: (?:{_esc(D['clock_words'])})()(?![{L}]))?"
    after_number = D.get("abbr_after_number", {})
    return {
        "L": L,
        "abbr": abbr,
        "abbr_after_number": re.compile(rf"(?<=[0-9])( ?)({_esc(after_number)})") if after_number else None,
        "initials": re.compile(rf"(?<![{L}])([{L}]+)\.(?= ?[{L}])"),
        "mobile": re.compile(rf"(\+91[ -]?)?(?<![0-9])([6-9][0-9]{{4}})[ -]?([0-9]{{5}})(?![0-9]){SUF}"),
        "landline": re.compile(rf"(?<![0-9])(0[0-9]{{2,4}})[ -]([0-9]{{6,8}})(?![0-9]){SUF}"),
        "pin": re.compile(rf"({_esc(D['pin_words'] + ['PIN', 'Pin', 'pin', 'Pincode', 'PIN code'])}|[{L}]+ ?-)"
                          rf" ?:? ?([0-9]{{6}})(?![0-9]){SUF}"),
        "iso": re.compile(rf"(?<![0-9])([0-9]{{4}})-([0-9]{{2}})-([0-9]{{2}})(?![0-9]){SUF}"),
        "dmy": re.compile(rf"(?<![0-9.])([0-9]{{1,2}})[/.-]([0-9]{{1,2}})[/.-]([0-9]{{4}})(?![0-9]){SUF}"),
        "month_year": re.compile(rf"(?<![A-Za-z{L}])({_esc(months)}),? ([0-9]{{4}})(?![0-9]){SUF}"),
        "time": re.compile(rf"(?<![0-9])([01]?[0-9]|2[0-3]):([0-5][0-9])(?::[0-5][0-9])?(?: ?([AaPp]\.?[Mm]\.?))?"
                           rf"(?![A-Za-z]){clock_words}{SUF}"),
        "time_ampm": re.compile(rf"(?<![0-9.,])(1[0-2]|0?[1-9]) ?([AaPp]\.?[Mm]\.?)(?![A-Za-z]){SUF}"),
        "money": re.compile(rf"({_esc(sum(cur.values(), []))}) ?({NUM})(?: ?({scale_alt})(?![A-Za-z{L}]))?"
                            rf"(?![0-9]){SUF}"),
        "currency_of": {s: k for k, v in cur.items() for s in v},
        "percent": re.compile(rf"({NUM}) ?%{SUF}"),
        "range": re.compile(rf"(?<![0-9.,])([0-9]+) ?[-–] ?([0-9]+)(?![0-9.,])(?: ?({units})(?![A-Za-z])| ?(%))?{SUF}"),
        "negative": re.compile(rf"(?<![0-9A-Za-z{L}])-(?=[0-9])"),
        "units": re.compile(rf"({NUM}) ?({units})(?![A-Za-z])"),
        "en_ordinal": re.compile(r"(?<![0-9])([0-9]+)(?:st|nd|rd|th)(?![A-Za-z])"),
        "long": re.compile(rf"(?<![.,0-9])(0[0-9]+|[0-9]{{7,}})(?![0-9]){SUF}"),
        "number": re.compile(rf"({NUM})(?![0-9]){SUF}"),
        "acronym": re.compile(r"(?<![A-Za-z])[A-Z]{2,6}(?![A-Za-z])"),
        "space_punct": re.compile(r" +([.,?!;:।])"),
        "spaces": re.compile(r" {2,}"),
        "year_before": re.compile(rf"(?:^|[^{L}])(?:{_esc(D['year_before'])}) $") if D["year_before"] else None,
        "year_after": tuple(D["year_after"]),
        "next_letter": re.compile(rf"^ [{L}]"),
        "months": tuple(months),
    }


# ------------------------------------------------------------------ rules
def _period(D, hour, ampm):
    P = D["periods"]
    if ampm == "am":
        return P["midnight"] if hour == 12 else P["night"] if hour <= 3 else P["morning"]
    if hour == 12 or hour <= 3:
        return P["noon"]
    return P["evening"] if hour <= 6 else P["night"]


def _clock(D, lang, hour, minute, ampm):
    if hour > 12:                       # 24-hour clock (an am/pm after it is ignored)
        ampm, hour = "pm", hour - 12
    elif hour == 0:
        ampm, hour = "am", 12
    p = _period(D, hour, ampm) if ampm else ""
    C = D["clock"]
    m = cardinal(minute, lang, True)
    if hour == 1 and "one" in C:
        tpl = C["one"][1 if minute else 0]
        h = ""
    else:
        tpl = C["hm"] if minute else C["h"]
        h = C["hours"][hour - 1] if "hours" in C else cardinal(hour, lang, True)
    return " ".join(tpl.replace("{p}", p).replace("{h}", h).replace("{m}", m).split())


def _date(D, lang, day, month, year, suffix):
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    words = f"{cardinal(day, lang)} {D['months'][month - 1]} {year_words(year, lang)}"
    return " " + attach(words, suffix, lang) + " "


def _suffix_words(raw, suffix, lang, attr, year):
    """A number with an attached suffix: ordinal, Bengali date day, or case ending."""
    D = DATA[lang]
    plain = "," not in raw and "." not in raw
    n = int(raw) if plain and len(raw) <= MAX_DIGITS else None
    if suffix and n is not None:
        day = D.get("day_suffixes", {}).get(suffix)
        if day and n in day[0]:
            form = day[1]
            if isinstance(form, dict):
                return form[str(n)]
            if form == "ই":
                return cardinal(n, lang) + form
            if len(day[0]) == 1:
                return form
            return attach(cardinal(n, lang), form, lang)
        O = D["ordinal"]["suffixes"]
        s = _sign_vowel(suffix[0]) + suffix[1:] if _is_sign(suffix[0]) else suffix
        hit = O.get(suffix) or O.get(s)
        if hit and (len(hit) == 1 or n in hit[1]):
            return ordinal(n, lang, hit[0])
    if year and n is not None:
        words = year_words(n, lang)
    else:
        words = number_text(raw, lang, attr or bool(suffix and D.get("suffix_attr")))
    return attach(words, suffix, lang)


def normalize_numbers(text, lang):
    if lang not in DATA:
        raise ValueError(f"no number data for {lang!r}")
    D = DATA[lang]
    P = _patterns(lang)
    text = unicodedata.normalize("NFC", text).translate(_DIGITS).translate(_SPACES)
    for rx, spoken in P["abbr"]:
        text = rx.sub(spoken, text)
    if P["abbr_after_number"]:
        text = P["abbr_after_number"].sub(lambda m: " " + D["abbr_after_number"][m[2]] + " ", text)
    letters = set(D["letters"])
    text = P["initials"].sub(lambda m: m[1] + " " if m[1] in letters else m[0], text)

    def mobile(m):
        plus = D["plus"] + " " + digits("91", lang) + " " if m[1] else ""
        return " " + plus + attach(digits(m[2] + m[3], lang), m[4], lang) + " "
    text = P["mobile"].sub(mobile, text)
    text = P["landline"].sub(lambda m: " " + attach(digits(m[1] + m[2], lang), m[3], lang) + " ", text)
    text = P["pin"].sub(lambda m: " " + m[1].rstrip("- ") + " " + attach(digits(m[2], lang), m[3], lang) + " ",
                        text)
    text = P["iso"].sub(lambda m: _date(D, lang, int(m[3]), int(m[2]), int(m[1]), m[4]) or m[0], text)
    text = P["dmy"].sub(lambda m: _date(D, lang, int(m[1]), int(m[2]), int(m[3]), m[4]) or m[0], text)
    text = P["month_year"].sub(lambda m: f" {m[1]} " + attach(year_words(int(m[2]), lang), m[3], lang) + " ", text)

    def time(m):
        ampm = (m[3] or "").replace(".", "").lower() or None
        return " " + attach(_clock(D, lang, int(m[1]), int(m[2]), ampm), m[5] or m[4], lang) + " "
    text = P["time"].sub(time, text)
    text = P["time_ampm"].sub(lambda m: " " + attach(_clock(D, lang, int(m[1]), 0, m[2].replace(".", "").lower()),
                                                     m[3], lang) + " ", text)

    def money(m):
        kind = P["currency_of"][m[1]]
        amount, scale, suffix = m[2], m[3], m[4]
        big = "cent" if kind != "inr" else "paise"
        if scale:
            exp = EN_SCALES.get(scale)
            if exp is None:
                exp = next(e for name, e in (("thousand", 3), ("lakh", 5), ("crore", 7))
                           if scale in D["scales"][name]["forms"])
            whole = _shift(amount, exp)
            if whole is not None:
                words = integer_words(whole, lang, True) + " " + _plural(D, kind, whole == "1")
            else:
                name = {3: "thousand", 5: "lakh", 7: "crore"}.get(exp)
                scale_word = _pick(D["scales"][name]["forms"], 2) if name else scale
                words = number_text(amount, lang) + " " + scale_word + " " + _plural(D, kind, False)
            return " " + attach(words, suffix, lang) + " "
        whole, _, frac = amount.replace(",", "").partition(".")
        whole = whole or "0"
        words = integer_words(whole, lang, True) + " " + _plural(D, kind, whole.lstrip("0") == "1")
        sub_units = int(frac.ljust(2, "0")[:2]) if frac else 0
        if sub_units:
            sep = " " + D["money_and"] + " " if D["money_and"] else " "
            words += sep + cardinal(sub_units, lang, True) + " " + _plural(D, big, sub_units == 1)
        return " " + attach(words, suffix, lang) + " "
    text = P["money"].sub(money, text)
    text = P["percent"].sub(lambda m: " " + attach(number_text(m[1], lang, True) + " " + D["percent"], m[2], lang)
                            + " ", text)

    def rng(m):
        a = attach(integer_words(m[1], lang), D["range"][0], lang)
        if D["range"][1]:
            a += " " + D["range"][1]
        b = integer_words(m[2], lang, bool(m[3] or m[4]))
        if m[3]:
            b += " " + _measure(D, m[3], False)
        elif m[4]:
            b += " " + D["percent"]
        return " " + a + " " + attach(b, m[5], lang) + " "
    text = P["range"].sub(rng, text)
    text = P["negative"].sub(" " + D["minus"] + " ", text)

    def unit(m):
        one = m[1].replace(",", "") == "1"
        return " " + number_text(m[1], lang, True) + " " + _measure(D, m[2], one) + " "
    text = P["units"].sub(unit, text)
    text = P["en_ordinal"].sub(lambda m: " " + ordinal(int(m[1]), lang) + " "
                               if len(m[1]) <= MAX_DIGITS else m[0], text)

    def long_digits(m):
        s = m[1]
        if s[0] != "0" and len(s) <= MAX_DIGITS and re.fullmatch(r"[0-9]+?0{5,}", s):
            return " " + attach(cardinal(int(s), lang), m[2], lang) + " "
        return " " + attach(digits(s, lang), m[2], lang) + " "
    text = P["long"].sub(long_digits, text)

    def number(m):
        raw, suffix = m[1], m[2] or ""
        suffix = suffix.replace("‌", "").replace("‍", "")
        before, after = m.string[:m.start()], m.string[m.end():]
        year = False
        if re.fullmatch(r"1[1-9][0-9]{2}", raw):
            yb, ya = P["year_before"], P["year_after"]
            nxt = after[1:].split(" ", 1)[0] if after.startswith(" ") else ""
            year = (bool(yb and yb.search(before)) or any(nxt.startswith(w) for w in ya)
                    or any(suffix.startswith(w) for w in ya) or nxt in P["months"])
        attr = not suffix and bool(P["next_letter"].match(after))
        return " " + _suffix_words(raw, suffix, lang, attr, year) + " "
    text = P["number"].sub(number, text)

    def acronym(m):
        w = m[0]
        if w in ACRONYM_KEEP:
            return w.lower()
        return " " + " ".join(D["letters"][ord(c) - 65] for c in w) + " "
    text = P["acronym"].sub(acronym, text)
    for sym, word in D["symbols"].items():
        text = text.replace(sym, " " + word + " ")
    text = text.replace("%", " " + D["percent"] + " ").replace("°", " " + _measure(D, "°", False) + " ")
    text = P["space_punct"].sub(r"\1", text)
    return P["spaces"].sub(" ", text).strip(" ")
