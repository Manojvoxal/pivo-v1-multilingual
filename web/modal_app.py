"""Deploy Voxal Studio on Modal.

Modal only serves static files: the multilingual PiVo model runs in each visitor's browser, on their
own CPU. There is no GPU and no inference server, so one small CPU container handles many
visitors and costs very little.

    python web/build.py               # makes web/dist
    modal deploy web/modal_app.py     # prints the public URL

Options (environment variables at deploy time):
    VOXAL_MIN_CONTAINERS=1   keep one container always warm (no cold start for the first
                             visitor after a quiet period; costs a little while idle)
"""

import os
from pathlib import Path

import modal

WEB = Path(__file__).resolve().parent
DIST = WEB / "dist"

if modal.is_local() and not (DIST / "manifest.json").exists():
    raise SystemExit("web/dist is missing. Run `python web/build.py` first.")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(WEB / "server.py", remote_path="/root/voxal_server.py")
    .add_local_dir(DIST, remote_path="/site")
)

# its own app + URL, so the Kannada + English site (pivo-v1, app "voxal-studio") keeps running
app = modal.App("voxal-studio-multi", image=image)


@app.function(
    cpu=0.5,
    memory=512,
    timeout=300,
    min_containers=int(os.environ.get("VOXAL_MIN_CONTAINERS", "0")),
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app(label="voxal-studio-multi")
def web():
    from voxal_server import StaticSite

    return StaticSite("/site")
