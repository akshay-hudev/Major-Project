"""Launch the zero-dependency browser demo for the project.

Usage:
    python3 demo_server.py
    python3 demo_server.py --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo"


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cardiac progression demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not (DEMO_DIR / "index.html").exists():
        raise SystemExit(f"Demo page is missing: {DEMO_DIR / 'index.html'}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DEMO_DIR)
    with ReusableTCPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{args.port}"
        print("\nHeart Disease Temporal Analysis — Demo")
        print(f"Open: {url}")
        print("Press Ctrl+C to stop.\n")
        if not args.no_browser:
            threading.Timer(0.6, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDemo stopped.")


if __name__ == "__main__":
    main()
