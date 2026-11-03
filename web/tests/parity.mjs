// Checks that web/app/frontend.js gives the same output as the Python frontend.
//   python web/tests/parity_ref.py web/tests/parity_ref.json
//   node web/tests/parity.mjs web/tests/parity_ref.json web/dist/lexicon/cmudict.txt
import { readFileSync } from "node:fs";
import { phonemize, setLexicon, splitUtterances, hasSpeech } from "../app/frontend.js";

const [refPath, lexPath] = process.argv.slice(2);
setLexicon(readFileSync(lexPath, "utf8"));
const cases = JSON.parse(readFileSync(refPath, "utf8"));

let failed = 0, lenientFailures = 0;
const show = (c, got) => {
  if (failed++ < 25) {
    console.log("MISMATCH", JSON.stringify(c.text));
    console.log("  python:", JSON.stringify({ n: c.normalized, e: c.error, u: c.unknown }));
    console.log("  js:    ", JSON.stringify({ n: got.normalized, e: got.error, u: got.unknown }));
    if (c.ids && got.ids) console.log("  ids py:", c.ids.join(","), "\n  ids js:", got.ids.join(","));
  }
};
const t0 = performance.now();
for (const c of cases) {
  let got;
  try {
    const r = phonemize(c.text);
    got = { normalized: r.normalized, ids: r.ids, unknown: r.unknown, error: null };
  } catch (e) {
    got = { error: e.kind ?? `js:${e.name}:${e.message}` };
  }
  const same = c.error || got.error
    ? c.error === got.error
    : c.normalized === got.normalized && c.ids.join() === got.ids.join() && c.unknown.join() === got.unknown.join();
  if (!same) show(c, got);

  // lenient mode (what the web app uses) must never throw, and must agree whenever strict mode succeeds
  try {
    const r = phonemize(c.text, { lenient: true });
    if (!c.error && (r.ids.join() !== c.ids.join())) { lenientFailures++; console.log("LENIENT DIFF", JSON.stringify(c.text)); }
    for (const u of splitUtterances(r.ids, 40)) {
      if (u[0] !== 1 || u[u.length - 1] !== 2 || u.length > 42 || !hasSpeech(u)) throw new Error("bad split " + u);
    }
  } catch (e) {
    lenientFailures++;
    if (lenientFailures < 10) console.log("LENIENT THROW", JSON.stringify(c.text), e.message);
  }
}
const ms = performance.now() - t0;
console.log(`${cases.length} cases, ${failed} mismatches, ${lenientFailures} lenient failures, ${(ms / cases.length / 2).toFixed(3)} ms/case`);
process.exit(failed || lenientFailures ? 1 : 0);
