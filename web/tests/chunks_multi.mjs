// Utterance token ids the web app would synthesize for a text in a language (used by e2e_multi.py).
//   node web/tests/chunks_multi.mjs web/dist/lexicon/cmudict.txt hi "नमस्ते"
import { readFileSync } from "node:fs";
import { setLexicon, splitUtterances } from "../app/frontend.js";
import { phonemize } from "../app/frontend_multi.js";

const [lexPath, lang, text] = process.argv.slice(2);
setLexicon(readFileSync(lexPath, "utf8"));
const front = phonemize(text, lang);
console.log(JSON.stringify({ normalized: front.normalized, chunks: splitUtterances(front.ids, 160) }));
