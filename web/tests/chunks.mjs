// Prints the utterance token ids the web app would synthesize for a text (used by e2e.py).
//   node web/tests/chunks.mjs web/dist/lexicon/cmudict.txt "ನಮಸ್ಕಾರ"
import { readFileSync } from "node:fs";
import { phonemize, setLexicon, splitUtterances } from "../app/frontend.js";

const [lexPath, text] = process.argv.slice(2);
setLexicon(readFileSync(lexPath, "utf8"));
const front = phonemize(text, { lenient: true });
console.log(JSON.stringify({ normalized: front.normalized, chunks: splitUtterances(front.ids, 160) }));
