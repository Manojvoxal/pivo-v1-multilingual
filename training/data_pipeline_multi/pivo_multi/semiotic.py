"""Shared pre-pass for reading numbers and symbols aloud (all 12 languages).

    prenormalize(text, lang, extra=None) -> text

Runs first inside every number reader (numnorm.py for hi mr gu pa bn or as ta te ml,
normalize_kn.py for Kannada and Indian English) and rewrites the structures those readers
do not know into forms they do, so the language-specific rules stay small. Classes follow
NVIDIA NeMo's text-normalization taxonomy (telephone, electronic, date, time, fraction,
measure, math, serial, roman, ...):

    telephone    020 7946 0018, (080) 2345 6789, +44 20 7946 0018, 1800-11-4000,
                 "Call 23456789"                       -> digit by digit, a pause per group
    electronic   info@example.com, www.example.com/a, @handle, #tag -> "at", "dot", "slash" ...
    date         15 Aug 2024, Aug 15, 2024, 2024/08/15, 15/08/24  -> 15/08/2024 (read by the
                 language's date rule); 15 Aug, Aug 2024 -> the language's month name
    time         6.15 pm -> 6:15 pm, 9am -> 9:00 am, 10:30:45 -> 10:30 and 45 seconds,
                 9am-5pm / 6:15-7:30 -> "<from> ... <to>" in the language's own way
    fraction     ½ ¼ ¾, 1/2, 2 1/2, 3/4                -> half / quarter / "a by b" patterns
    measure      5'11"                                 -> 5 feet 11 inches; 10K -> 10000
    math         × ÷ ^ ± < > * ² ³ ~                   -> times, divided by, power ...
    serial       COVID-19, AI-202, KA-01-AB-1234, 4G, H2O -> letters and numbers apart
    dotted       3.2.1, 192.168.1.1                    -> "3 dot 2 dot 1"
    ratio        16:9                                  -> "16 is to 9" pattern
    decade       1990s                                 -> the language's decade phrase
    roman        World War II, Chapter IV, Elizabeth II -> 2, 4, the 2nd
    symbols      #5 / № 5 -> number 5, © -> copyright, ™ ® dropped, "/-" after an amount dropped

Words for each language are in semiotic.json. The caller passes what it already owns
(digit words, month names, point / plus words, range joiner) in `extra`.
web/app/semiotic.js is a line-by-line port: keep the two in step (web/tests/numbers_parity.*).
"""

import json
import re
import unicodedata
from pathlib import Path

_HERE = Path(__file__).parent        # pivo_multi/semiotic.json, or src/pivo/resources/semiotic.json
WORDS = json.loads(next(p for p in (_HERE / "semiotic.json", _HERE / "resources" / "semiotic.json")
                        if p.exists()).read_text(encoding="utf-8"))

EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
MONTH_INDEX = {}
for _i, _m in enumerate(EN_MONTHS):
    for _form in (_m, _m.upper(), _m[:3], _m[:3].upper()):
        MONTH_INDEX[_form] = _i + 1
MONTH_INDEX["Sept"] = MONTH_INDEX["SEPT"] = 9
_MON = "|".join(sorted(MONTH_INDEX, key=lambda w: (-len(w), w)))

VULGAR = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5",
          "⅘": "4/5", "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
SUPERSCRIPT = {"⁰": "0", "¹": "1", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
         "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19,
         "XX": 20}
ROMAN_KEYWORDS = ["Chapter", "Part", "Volume", "Vol.", "Class", "Phase", "Section", "Article", "Schedule", "Book",
                  "Act", "Grade", "Standard", "Std.", "Plan", "Round", "Level", "Season", "World War", "Edition",
                  "Unit", "Paper", "Term", "Episode", "Stage", "Tier", "Division", "Rule", "Clause"]
LATIN_ABBR = {"ext.": "extension", "Ext.": "extension", "e.g.": "for example", "i.e.": "that is",
              "approx.": "approximately", "Ltd.": "limited", "Pvt.": "private", "Govt.": "government", "Jr.": "junior", "Sr.": "senior", "Dept.": "department",
              "Rd.": "road", "Estd.": "established", "Max.": "maximum", "Min.": "minimum"}
PHONE_WORDS_EN = ["phone", "Phone", "PHONE", "ph", "Ph", "mob", "Mob", "mobile", "Mobile", "call", "Call",
                  "tel", "Tel", "telephone", "Telephone", "contact", "Contact", "dial", "Dial", "fax", "Fax",
                  "helpline", "Helpline", "whatsapp", "WhatsApp", "Whatsapp", "cell", "Cell", "landline",
                  "Landline", "toll free", "Toll free", "Toll Free", "toll-free", "Toll-free"]
# letters right after a number that are a unit, an ordinal or a scale, not a serial code
NUMBER_TAILS = {"st", "nd", "rd", "th", "am", "pm", "AM", "PM", "K", "k", "s", "x", "X", "L", "l", "m", "g", "W",
                "V", "km", "kg", "cm", "mm", "ml", "mL", "gm", "ltr", "Ltr", "mg", "hr", "hrs", "Hrs", "min",
                "mins", "sec", "secs", "ms", "GB", "MB", "KB", "TB", "MP", "Hz", "kHz", "MHz", "GHz", "kW",
                "kWh", "mAh", "Mbps", "Gbps", "mph", "kmph", "ft", "lb", "lbs", "oz", "ha", "yr", "yrs", "kms",
                "Km", "KM", "Kg", "KG", "cr", "Cr", "mn", "bn", "lakh", "lakhs", "crore", "crores", "C", "F"}

_DIGITS = {}
for _base in range(0x900, 0xD80, 0x80):
    for _d in range(10):
        _DIGITS[_base + 0x66 + _d] = str(_d)
_DROP = re.compile("[\u2122\u00ae\ufe0f\ufe0e\u20e3\u200b]")      # ™ ® and invisible marks (emoji stay: callers list them)


def _words(lang, extra):
    W = dict(WORDS[lang])
    if extra:
        W.update(extra)
    return W


def _digits(s, W):
    return " ".join(W["digits"][int(c)] for c in s if c.isascii() and c.isdigit())


def _pick(form, one):
    if isinstance(form, str):
        return form
    return form[0] if one else form[1]


def _phone(groups, W, plus=False):
    spoken = ", ".join(_digits(g, W) for g in groups if g)
    return " " + (W["plus"] + " " if plus else "") + spoken + " "


def _date(day, month, year):
    if 1 <= month <= 12 and 1 <= day <= 31:
        return f"{day:02d}/{month:02d}/{year}"
    return None


def _fill(template, **parts):
    out = template
    for key, value in parts.items():
        out = out.replace("{" + key + "}", value)
    return out


def _mer(token):
    return token.replace(".", "").replace(" ", "").lower()


def _roman_value(token):
    return ROMAN.get(token)


def _ordinal_digits(n):
    """2 -> '2nd' (the language's ordinal rule reads it)."""
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def prenormalize(text, lang, extra=None):
    W = _words(lang, extra)
    en = lang == "en"
    L = "A-Za-z\u0900-\u0D7F"                       # letters of every supported script
    text = unicodedata.normalize("NFC", text).translate(_DIGITS)
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")

    # ---- symbols that are never read, and money's "/-"
    text = _DROP.sub("", text)
    text = re.sub(r"(?:(?<![A-Za-z])[Cc]opyright\s?)?©", " " + W["copyright"] + " ", text)
    text = re.sub(r"(?<=[0-9])\s?/-", "", text)
    text = text.replace("\\", " ")
    # ---- dashes and minus between numbers -> "-"
    text = re.sub(r"(?<=[0-9])\s?[\u2013\u2014\u2012]\s?(?=[0-9])", "-", text)
    text = text.replace("\u2212", "-")
    # ---- vulgar fractions: 1½ -> 1 1/2
    for ch, frac in VULGAR.items():
        text = re.sub(rf"(?<=[0-9]){ch}", " " + frac, text)
        text = text.replace(ch, " " + frac + " ")

    # ---- electronic: e-mail, URL, bare domain, @handle, #tag
    def spell_part(part):
        out = []
        for piece in re.split(r"([._\-/:])", part):
            if piece in (".", "_", "-", "/", ":"):
                out.append({".": W["dot"], "_": W["underscore"], "-": W["hyphen"], "/": W["slash"],
                            ":": ""}[piece])
            elif piece.isdigit():
                out.append(_digits(piece, W))
            elif piece:
                out.append(piece)
        return " ".join(w for w in out if w)

    def email(m):
        return " " + spell_part(m[1]) + " " + W["at"] + " " + spell_part(m[2]) + " "
    text = re.sub(r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?![A-Za-z0-9])",
                  email, text)

    def url(m):
        body = m[2].rstrip("/")
        host, _, path = body.partition("/")
        words = []
        if host.lower().startswith("www."):
            words.append("W W W")
            host = host[4:]
        words.append(spell_part(host))
        if path:
            words.append(W["slash"] + " " + spell_part(path))
        return m[1] + " ".join(words) + " "
    tlds = "com|org|net|in|co|io|ai|gov|edu|info|app|dev|uk|us|me|tv|biz|ac|nic|res"
    text = re.sub(rf"(^|[\s(])(?:https?://)?((?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:{tlds})"
                  rf"(?:\.(?:in|uk|au))?(?:/[A-Za-z0-9._~/-]*)?)(?=[\s,;:!?)]|\.(?:\s|$)|$)", url, text)
    text = re.sub(r"(?<![A-Za-z0-9])https?://", "", text)
    text = re.sub(r"(^|\s)@([A-Za-z_][A-Za-z0-9_]*)", lambda m: m[1] + W["at"] + " " + m[2].replace("_", " "), text)
    text = re.sub(r"(^|\s)#([A-Za-z][A-Za-z0-9_]*)", lambda m: m[1] + W["hashtag"] + " " + m[2].replace("_", " "),
                  text)
    # number signs: #5, № 5, No. 5
    text = re.sub(r"(?:#|№|(?<![A-Za-z0-9_])No\.|(?<![A-Za-z0-9_])NO\.)\s?(?=[0-9])", " " + W["number"] + " ", text)

    # ---- dates
    def mon(token):
        return MONTH_INDEX.get(token.rstrip("."))
    ordsuf = r"(?:st|nd|rd|th)?"
    text = re.sub(rf"(?<![0-9])([0-9]{{1,2}}){ordsuf}\s?(?:of\s)?({_MON})\.?,?\s([0-9]{{4}})(?![0-9])",
                  lambda m: _date(int(m[1]), mon(m[2]), int(m[3])) or m[0], text)
    text = re.sub(rf"(?<![A-Za-z])({_MON})\.?\s([0-9]{{1,2}}){ordsuf},?\s([0-9]{{4}})(?![0-9])",
                  lambda m: _date(int(m[2]), mon(m[1]), int(m[3])) or m[0], text)
    text = re.sub(r"(?<![0-9./-])([0-9]{1,2})[/.-]([0-9]{1,2})[/.-]([0-9]{4})(?![0-9]|[.,/-][0-9])",
                  lambda m: _date(int(m[1]), int(m[2]), int(m[3])) or m[0], text)
    text = re.sub(r"(?<![0-9./-])([0-9]{4})([/.-])([0-9]{1,2})\2([0-9]{1,2})(?![0-9]|[.,/-][0-9])",
                  lambda m: _date(int(m[4]), int(m[3]), int(m[1])) or m[0], text)
    text = re.sub(r"(?<![0-9./-])([0-9]{4})[/.]([0-9]{1,2})[/.]([0-9]{1,2})(?![0-9])",
                  lambda m: _date(int(m[3]), int(m[2]), int(m[1])) or m[0], text)
    text = re.sub(r"(?<![0-9./-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2})(?![0-9]|[.,/-][0-9])",
                  lambda m: _date(int(m[1]), int(m[3]), (2000 if int(m[4]) <= 49 else 1900) + int(m[4])) or m[0],
                  text)
    months = W["months"]

    def month_name(n):
        return EN_MONTHS[n - 1] if en else months[n - 1]
    text = re.sub(rf"(?<![0-9])([0-9]{{1,2}})({ordsuf})\s?(?:of\s)?({_MON})\.?(?![A-Za-z])",
                  lambda m: f"{m[1]}{m[2] if en else ''} {month_name(mon(m[3]))}" if 1 <= int(m[1]) <= 31
                  else m[0], text)
    text = re.sub(rf"(?<![A-Za-z])({_MON})\.?\s([0-9]{{1,2}})({ordsuf})(?![0-9])",
                  lambda m: (f"{month_name(mon(m[1]))} {m[2]}{m[3]}" if en else f"{m[2]} {month_name(mon(m[1]))}")
                  if 1 <= int(m[2]) <= 31 else m[0], text)
    text = re.sub(rf"(?<![A-Za-z])({_MON})\.?,?\s(?=[0-9]{{4}}(?![0-9]))",
                  lambda m: month_name(mon(m[1])) + " ", text)

    # ---- telephone numbers
    text = re.sub(r"(?<![0-9+])\+(?!91[\s-]?[6-9][0-9]{4}[\s-]?[0-9]{5}(?![0-9]))([0-9]{1,3})"
                  r"((?:(?:[\s-]+|(?<=\)))[0-9]{1,5}|[\s-]*\([0-9]{1,5}\)){2,6})(?![0-9])",
                  lambda m: _phone([m[1]] + re.findall(r"[0-9]+", m[2]), W, plus=True)
                  if len(re.sub(r"[^0-9]", "", m[0])) >= 8 else m[0], text)
    text = re.sub(r"(?<![0-9])\((0?[0-9]{2,5})\)[\s-]*([0-9]{2,5})(?:[\s-]([0-9]{2,5}))?(?![0-9])",
                  lambda m: _phone([m[1], m[2], m[3]], W), text)
    text = re.sub(r"(?<![0-9.,])(0[0-9]{1,4})((?:[\s-][0-9]{2,5}){2,3})(?![0-9]|[.,][0-9])",
                  lambda m: _phone([m[1]] + re.findall(r"[0-9]+", m[2]), W), text)
    text = re.sub(r"(?<![0-9.,])(1[89]00|1860)[\s-]?([0-9]{2,4})[\s-]?([0-9]{3,4})(?![0-9])",
                  lambda m: _phone([m[1], m[2], m[3]], W), text)
    text = re.sub(r"(?<![0-9.,])([0-9]{3})-([0-9]{3})-([0-9]{4})(?![0-9])",
                  lambda m: _phone([m[1], m[2], m[3]], W), text)
    phone_words = PHONE_WORDS_EN + W.get("phone_words", [])
    alt = "|".join(re.escape(w) for w in sorted(phone_words, key=lambda w: (-len(w), w)))
    text = re.sub(rf"((?<![{L}])(?:{alt})(?![{L}])\.?(?:\s?(?:[Nn]o\.?|[Nn]umber|#))?\s?:?\s?)"
                  r"(?<![0-9])([0-9]{2,5}(?:[\s-][0-9]{2,5}){0,3})(?![0-9]|[.,][0-9])",
                  lambda m: m[1] + _phone(re.findall(r"[0-9]+", m[2]), W)
                  if len(re.sub(r"[^0-9]", "", m[2])) >= 6
                  and not re.fullmatch(r"[6-9][0-9]{4}[\s-]?[0-9]{5}", m[2]) else m[0], text)   # mobiles: reader

    # ---- times
    # am / pm in one spelling ("p.m." loses its dots; a sentence-final "pm." keeps the full stop)
    ampm = r"[AaPp](?:\.\s?[Mm]\.|\s?[Mm](?![A-Za-z]))"
    text = re.sub(rf"(?<![0-9:])(?<![0-9]\.)([01]?[0-9]|2[0-3])[.:]([0-5][0-9])\s?({ampm})([A-Za-z]?)",                 # "p.m.km": keep a space
                  lambda m: f"{m[1]}:{m[2]} {_mer(m[3])}" + (" " + m[4] if m[4] else ""), text)
    text = re.sub(rf"(?<![0-9:])(?<![0-9][.,])(1[0-2]|0?[1-9])\s?({ampm})([A-Za-z]?)",
                  lambda m: f"{m[1]}:00 {_mer(m[2])}" + (" " + m[3] if m[3] else ""), text)
    if en:
        text = re.sub(r"(?<![0-9:])12(?::00)?\s?(noon|midnight)(?![A-Za-z0-9_])", r"12 \1", text)
    else:
        text = re.sub(r"(?<![0-9:])12(?::00)?\s?noon(?![A-Za-z0-9_])", "12:00 pm", text)
        text = re.sub(r"(?<![0-9:])12(?::00)?\s?midnight(?![A-Za-z0-9_])", "12:00 am", text)

    def seconds(m):
        hour, minute, sec, mer = m[1], m[2], int(m[3]), m[4] and _mer(m[4])
        tail = (" " + mer) if mer else ""
        if not sec:
            return f"{hour}:{minute}{tail}"
        conj = (W["and"] + " ") if W["and"] else ""
        return f"{hour}:{minute}{tail} {conj}{sec} {_pick(W['seconds'], sec == 1)}"
    text = re.sub(rf"(?<![0-9:])([01]?[0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])(?:\s?({ampm}))?(?![0-9])",
                  seconds, text)
    clock = rf"(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?:\s?{ampm})?"
    rng = W["range"]

    def time_range(m):
        first = m[1].rstrip()
        joined = first + rng[0] + (" " + rng[1] if rng[1] else "")
        return joined + " " + m[2]
    text = re.sub(rf"(?<![0-9:])({clock})\s?(?:-|to|till|until)\s?({clock})(?![0-9])", time_range, text)

    # ---- decades (English handles its own)
    if not en:
        text = re.sub(r"(?<![0-9])(1[0-9][0-9]0|20[0-9]0)(?:'s|’s|s)(?![A-Za-z])",
                      lambda m: _fill(W["decade"], y=m[1]), text)

    # ---- roman numerals
    kw = "|".join(re.escape(k) for k in sorted(ROMAN_KEYWORDS, key=lambda w: (-len(w), w)))
    text = re.sub(rf"(?<![A-Za-z])({kw})\s(X{{0,2}}(?:IX|IV|V?I{{0,3}}))(?![A-Za-z])",
                  lambda m: f"{m[1]} {_roman_value(m[2])}" if _roman_value(m[2]) else m[0], text)
    text = re.sub(r"(?<![A-Za-z])([A-Z][a-z]{2,})\s(II|III|IV|VI|VII|VIII|IX|XI|XII|XIII|XIV|XV)(?![A-Za-z])",
                  lambda m: f"{m[1]} {'the ' if en else ''}{_ordinal_digits(ROMAN[m[2]])}", text)

    # ---- dotted numbers: versions, sections, IP addresses
    text = re.sub(r"(?<![0-9.])([0-9]+(?:\.[0-9]+){2,})(?![0-9]|\.[0-9])",
                  lambda m: " " + (" " + W["dot"] + " ").join(m[1].split(".")) + " ", text)
    text = re.sub(r"(?<![^\s(])\.([0-9])", r"0.\1", text)                       # " .5" -> " 0.5"

    # ---- serial codes: letters and digits apart (COVID-19, AI-202, KA-01-AB-1234, 4G, H2O)
    def serial(m):
        token = m[0]
        if not (re.search(r"[A-Za-z]", token) and re.search(r"[0-9]", token)):
            return token
        tail = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)", token)
        if tail and tail[2] in NUMBER_TAILS:
            return token
        if re.fullmatch(r"[0-9]+(?:st|nd|rd|th)", token) or re.search(r"[A-Za-z]\.", token):
            return token                                                  # 21st, "Prof.2", "p.m.8"
        # split only where letters meet digits; "13-12" keeps its dash, "51st" stays an ordinal
        return re.sub(r"(?<=[A-Za-z])-?(?=[0-9])|(?<=[0-9])-?(?=[A-Za-z])(?!(?:st|nd|rd|th)(?![a-z]))", " ", token)
    text = re.sub(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:\.[0-9]+)?(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9])", serial, text)

    # ---- scale letters after numbers: 10K -> 10000
    def thousands(m):
        whole, _, frac = m[1].partition(".")
        if len(frac) > 3:
            return m[0]
        return str(int(whole + frac.ljust(3, "0")))
    text = re.sub(r"(?<![0-9.])([0-9]{1,4}(?:\.[0-9]+)?)[Kk](?![A-Za-z0-9])", thousands, text)

    # ---- feet and inches: 5'11", 5 ft 11 in
    text = re.sub(r"(?<![0-9])([0-9]{1,2})\s?['’]\s?([0-9]{1,2})\s?(?:\"|”|'')",
                  lambda m: f"{m[1]} {W['feet']} {m[2]} {W['inches']}", text)
    text = re.sub(r"(?<![0-9])([0-9]{1,2})\s?(?:ft|feet)\s([0-9]{1,2})\s?(?:in|inch|inches)(?![A-Za-z])",
                  lambda m: f"{m[1]} {W['feet']} {m[2]} {W['inches']}", text)

    # ---- fractions (English reads its own) and ratios
    if not en:
        special = {"1/2": W["half"], "1/4": W["quarter"], "3/4": W["three_quarters"]}

        def fraction(m):
            a, b = m[2], m[3]
            words = special.get(f"{a}/{b}") or _fill(W["fraction"], a=a, b=b)
            if m[1]:
                conj = (W["and"] + " ") if W["and"] else ""
                return f"{m[1]} {conj}{words}"
            return words
        text = re.sub(r"(?<![0-9/.])(?:([0-9]{1,3})\s)?([1-9][0-9]{0,2}(?:\.[0-9]+)?)/([1-9][0-9]{0,2})"
                      r"(?![0-9/]|[.-][0-9])",
                      fraction, text)
    text = re.sub(r"(?<![0-9:.])([0-9]{1,3})(?::([0-9])|\s:\s([0-9]{1,3}))(?![0-9:])",
                  lambda m: " " + _fill(W["ratio"], a=m[1], b=m[2] or m[3]) + " ", text)

    # ---- math
    text = re.sub(r"(?<=[0-9])\s?[×xX*]\s?(?=[0-9])", " " + W["times"] + " ", text)
    text = re.sub(r"\s?×\s?", " " + W["times"] + " ", text)
    text = re.sub(r"\s?÷\s?", " " + W["divided_by"] + " ", text)
    text = re.sub(r"(?<=[0-9A-Za-z])\s?\^\s?(?=[0-9A-Za-z])", " " + W["power"] + " ", text)
    text = text.replace("±", " " + W["plus_minus"] + " ")
    text = re.sub(r"(?<=[0-9A-Za-z])\s?<\s?(?=[0-9A-Za-z])", " " + W["less_than"] + " ", text)
    text = re.sub(r"(?<=[0-9A-Za-z])\s?>\s?(?=[0-9A-Za-z])", " " + W["greater_than"] + " ", text)
    if en:
        text = re.sub(r"\s?=\s?", " " + W["equals"] + " ", text)
    text = re.sub(r"[~≈]\s?(?=[0-9])", W["about"] + " ", text)
    text = _squares(text, W)
    text = re.sub(r"([0-9A-Za-z])([⁰¹⁴⁵⁶⁷⁸⁹]+)",
                  lambda m: m[1] + " " + W["power"] + " " + "".join(SUPERSCRIPT[c] for c in m[2]) + " ", text)

    # ---- Latin abbreviations used in Indian English and mixed text
    for short, full in LATIN_ABBR.items():
        text = re.sub(rf"(?<![A-Za-z]){re.escape(short)}", " " + full + " ", text)
    text = re.sub(r"(?<![A-Za-z])etc\.?(?![A-Za-z])", " " + W["etc"] + " ", text)
    # word/word -> "word slash word" (Yes/No, and/or); "60 km/h" is a unit and stays
    text = re.sub(r"(?<![0-9A-Za-z\u0900-\u0D7F])(?<![0-9]\s)([A-Za-z\u0900-\u0D7F]+)/(?=[A-Za-z\u0900-\u0D7F])",
                  lambda m: m[1] + " " + W["slash"] + " ", text)
    return text


def _squares(text, W):
    """x² -> x squared, 5³ -> 5 cubed; m² / km² / ft³ stay (they are units)."""
    out = []
    for i, ch in enumerate(text):
        if ch in "²³":
            k = i
            while k > 0 and text[k - 1].isascii() and text[k - 1].isalpha():
                k -= 1
            if text[k:i] in ("km", "cm", "mm", "ft", "m") and (k == 0 or not text[k - 1].isalpha()):
                out.append(ch)
                continue
            out.append(" " + (W["squared"] if ch == "²" else W["cubed"]) + " ")
            continue
        out.append(ch)
    return "".join(out)
