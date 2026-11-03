"""Static file server for the Voxal Studio site in web/dist (run `python web/build.py` first).

The voice model runs in the visitor's browser, so this only serves files. It is a plain ASGI
app with no dependencies: files are held in memory, gzip copies made by the build are sent to
browsers that accept them, big files are cached for a year (their URLs carry a content hash),
everything else is revalidated with an ETag.

    pip install uvicorn
    python web/server.py                  # http://localhost:8000
    python web/server.py --port 9000 --host 0.0.0.0

On Modal the same app is served by web/modal_app.py.
"""

import argparse
import hashlib
from pathlib import Path

TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml", ".json": "application/json", ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".png": "image/png", ".ico": "image/x-icon",
}
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; media-src 'self' blob:; "
       "connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
COMMON_HEADERS = [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"strict-origin-when-cross-origin"),
                  (b"cross-origin-opener-policy", b"same-origin")]
CHUNK = 256 * 1024


class StaticSite:
    def __init__(self, root):
        root = Path(root)
        if not (root / "index.html").exists():
            raise FileNotFoundError(f"{root}/index.html not found; run `python web/build.py` first")
        self.files = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix == ".gz" or path.name.startswith("."):
                continue
            data = path.read_bytes()
            gz = path.with_name(path.name + ".gz")
            tag = hashlib.sha256(data).hexdigest()[:24]
            self.files["/" + path.relative_to(root).as_posix()] = {
                "data": data, "gz": gz.read_bytes() if gz.exists() else None,
                "type": TYPES.get(path.suffix, "application/octet-stream"),
                "etag": f'"{tag}"'.encode(), "etag_gz": f'"{tag}-gz"'.encode(),
                "csp": path.suffix in (".html", ".js", ".mjs"),
            }
        self.files["/"] = self.files["/index.html"]

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        method = scope["method"]
        if method not in ("GET", "HEAD"):
            return await self.respond(send, 405, b"Method not allowed", [(b"allow", b"GET, HEAD")])
        entry = self.files.get(scope["path"])
        if entry is None:
            return await self.respond(send, 404, b"Not found")

        headers = dict(scope.get("headers") or [])
        use_gz = entry["gz"] is not None and b"gzip" in headers.get(b"accept-encoding", b"")
        body, etag = (entry["gz"], entry["etag_gz"]) if use_gz else (entry["data"], entry["etag"])
        versioned = b"v=" in scope.get("query_string", b"")
        out = [(b"content-type", entry["type"].encode()), (b"etag", etag),
               (b"cache-control", b"public, max-age=31536000, immutable" if versioned else b"no-cache"),
               *COMMON_HEADERS]
        if entry["gz"] is not None:
            out.append((b"vary", b"accept-encoding"))
        if use_gz:
            out.append((b"content-encoding", b"gzip"))
        if entry["csp"]:
            out.append((b"content-security-policy", CSP.encode()))
        if etag in [t.strip() for t in headers.get(b"if-none-match", b"").split(b",")]:
            return await self.respond(send, 304, b"", out, with_type=False)
        out.append((b"content-length", str(len(body)).encode()))
        await send({"type": "http.response.start", "status": 200, "headers": out})
        if method == "HEAD":
            return await send({"type": "http.response.body", "body": b""})
        view = memoryview(body)
        for start in range(0, len(body), CHUNK):
            await send({"type": "http.response.body", "body": bytes(view[start:start + CHUNK]),
                        "more_body": start + CHUNK < len(body)})
        if not body:
            await send({"type": "http.response.body", "body": b""})

    @staticmethod
    async def respond(send, status, body, headers=(), with_type=True):
        out = list(headers)
        if with_type:
            out += [(b"content-type", b"text/plain; charset=utf-8"), *COMMON_HEADERS]
        if status != 304:
            out.append((b"content-length", str(len(body)).encode()))
        await send({"type": "http.response.start", "status": status, "headers": out})
        await send({"type": "http.response.body", "body": body})


def main():
    parser = argparse.ArgumentParser(description="Serve the Voxal Studio site locally.")
    parser.add_argument("--dist", type=Path, default=Path(__file__).resolve().parent / "dist")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is needed to run the server locally: pip install uvicorn")
    print(f"Voxal Studio: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    uvicorn.run(StaticSite(args.dist), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
