"""Command line synthesis, model download, and local HTTP serving."""
import argparse
import json
import os
from pathlib import Path

from .engine import PiVo, verify_model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pivo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthesize", help="generate a mono PCM16 WAV")
    synth.add_argument("--model-dir", default=os.getenv("PIVO_MODEL_DIR", "."))
    synth.add_argument("--text", required=True)
    synth.add_argument("--voice", default=None,
                       help="voice name (list them with `pivo voices`); default: the model's default voice")
    synth.add_argument("--speed", type=float, default=1.0)
    synth.add_argument("--threads", type=int, default=1)
    synth.add_argument("--overrides", type=Path, help="JSON word-to-ARPAbet-list mapping")
    synth.add_argument("--output", type=Path, required=True)
    voices = commands.add_parser("voices", help="list the voices of a model")
    voices.add_argument("--model-dir", default=os.getenv("PIVO_MODEL_DIR", "."))
    serve = commands.add_parser("serve", help="serve the buffered WAV HTTP API")
    serve.add_argument("--model-dir", default=os.getenv("PIVO_MODEL_DIR", "."))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--threads", type=int, default=1)
    download = commands.add_parser("download", help="download a pinned Hugging Face model revision")
    download.add_argument("--repo", required=True)
    download.add_argument("--revision", required=True, help="published model commit SHA")
    download.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "synthesize":
            overrides = json.loads(args.overrides.read_text()) if args.overrides else None
            model = PiVo(args.model_dir, threads=args.threads)
            print(model.save(args.output, args.text, voice=args.voice, speed=args.speed, overrides=overrides))
        elif args.command == "voices":
            model = PiVo(args.model_dir)
            for name, info in model.voice_info.items():
                default = "  (default)" if name == model.default_voice else ""
                print(f"{name}\t{info.get('language', 'en')}\t{info.get('speaker', '')}{default}")
        elif args.command == "serve":
            try:
                import uvicorn
                from .api import create_app
            except ImportError as exc:
                raise ValueError("install the server extra: pip install '.[server]'") from exc
            model = PiVo(args.model_dir, threads=args.threads)
            uvicorn.run(create_app(model), host=args.host, port=args.port)
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ValueError("install the hub extra: pip install '.[hub]'") from exc
            snapshot_download(repo_id=args.repo, revision=args.revision,
                              local_dir=str(args.output),
                              allow_patterns=["onnx/*", "model_manifest.json", "PROVENANCE.json", "LICENSE", "LICENSES/*", "NOTICE", "THIRD_PARTY.md", "README.md"])
            verify_model(args.output)
            print(args.output)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"pivo: {exc}\n")


if __name__ == "__main__":
    main()
