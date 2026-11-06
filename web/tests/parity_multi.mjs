// JavaScript multilingual frontend vs Python: node web/tests/parity_multi.mjs ref.json lexicon.txt
import { readFileSync } from "node:fs";
import { setLexicon } from "../app/frontend.js";
import { wordToPhones, phonemize } from "../app/frontend_multi.js";

const [refPath, lexPath] = process.argv.slice(2);
const ref = JSON.parse(readFileSync(refPath, "utf8"));
setLexicon(readFileSync(lexPath, "utf8"));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
let bad = 0;
const report = (kind, item, got) => { if (bad++ < 15) console.log(`MISMATCH ${kind}`, JSON.stringify(item), "-> JS", JSON.stringify(got)); };

let nw = 0;
for (const w of ref.words) {
  const got = wordToPhones(w.word, w.lang);
  nw++;
  if (!same(got, w.phones)) report("word", w, got);
}
let ns = 0;
for (const s of [...ref.sentences, ...ref.real]) {
  if (s.ids === null) continue;
  ns++;
  const got = phonemize(s.text, s.lang, { numbers: false });
  if (!same(got.ids, s.ids)) report("sentence", s, { ids: got.ids, skipped: got.skipped, unknown: got.unknown });
}
console.log(`${nw} words + ${ns} sentences (${ref.real.length} real) checked: ${bad ? bad + " MISMATCHES" : "all identical"}`);
process.exit(bad ? 1 : 0);
