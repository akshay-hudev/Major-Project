"""CardioPulse AI — Application Launcher

Launches the FastAPI backend and clinical intelligence frontend.

Usage:
    python run_app.py
    py -3.14 run_app.py
    python run_app.py --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Force UTF-8 stdout/stderr for Windows console
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


import socket

def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
    port = start_port
    while is_port_in_use(host, port) and port < start_port + max_attempts:
        port += 1
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CardioPulse AI Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Check port availability and auto-fallback if already occupied
    port = args.port
    if is_port_in_use(args.host, port):
        fallback_port = find_available_port(args.host, port + 1)
        print(f"\n[Notice] Port {port} is currently in use. Automatically switching to port {fallback_port}.\n")
        port = fallback_port

    import uvicorn
    from backend.inference_engine import get_inference_engine

    print("=" * 72)
    print("  CardioPulse AI — Longitudinal Cardiac Progression Platform")
    print("=" * 72)
    print("  Initializing clinical inference engine and models...")
    engine = get_inference_engine()
    print(f"  Active Model: {engine.model_name}")
    print(f"  Calibrated Decision Threshold: {engine.default_threshold:.3f}")
    print(f"  Temporal Sequence: 6 ICU Visits x 59 Features (826 engineered features)")
    print("-" * 72)

    url = f"http://{args.host}:{port}"
    docs_url = f"{url}/docs"
    print(f"  Frontend Dashboard:   {url}")
    print(f"  Interactive API Docs: {docs_url}")
    print("=" * 72)
    print("  Press Ctrl+C to stop the server.\n")

    if not args.no_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
