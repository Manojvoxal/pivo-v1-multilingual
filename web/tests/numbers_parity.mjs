// Number normaliser parity: app/numbers_multi.js must give exactly the text that the Python
// normaliser (training/data_pipeline_multi/pivo_multi/numnorm.py) gives.
//
//   python web/tests/numbers_parity_ref.py ref.json
//   node web/tests/numbers_parity.mjs ref.json
import { readFileSync } from "node:fs";
import { normalizeNumbers } from "../app/numbers_multi.js";

const rows = JSON.parse(readFileSync(process.argv[2], "utf8"));
const bad = {};
let fails = 0;
for (const r of rows) {
  let got;
  try { got = normalizeNumbers(r.text, r.lang); } catch (e) { got = `ERROR ${e.message}`; }
  if (got !== r.out) {
    fails++;
    (bad[r.lang] ||= []).push(r);
    if (fails <= 15) console.log(`MISMATCH ${r.lang} ${JSON.stringify(r.text)}\n  py: ${r.out}\n  js: ${got}`);
  }
}
const per = Object.fromEntries(Object.entries(bad).map(([k, v]) => [k, v.length]));
console.log(`${rows.length - fails}/${rows.length} identical`, fails ? JSON.stringify(per) : "");
process.exit(fails ? 1 : 0);
