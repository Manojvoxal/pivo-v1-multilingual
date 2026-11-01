"""Synthesize using the local HTTP server (standard library only)."""
import argparse
import json
import os
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--text", default="NASA is ready for the next mission.")
parser.add_argument("--voice", choices=["Pi", "Vo"], default="Pi")
parser.add_argument("--output", type=Path, default=Path("speech.wav"))
args = parser.parse_args()
headers = {"Content-Type": "application/json"}
if os.getenv("PIVO_API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["PIVO_API_KEY"]
request = urllib.request.Request(args.url.rstrip("/") + "/v1/audio/speech",
    data=json.dumps({"input": args.text, "voice": args.voice}).encode(), headers=headers)
with urllib.request.urlopen(request, timeout=120) as response:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.read())
print(args.output)
