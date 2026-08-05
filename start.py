#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import threading
import webbrowser


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


def parse_args():
    parser = argparse.ArgumentParser(description="Start the Medical Agent web app.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Flask bind host.")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Flask bind port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    return parser.parse_args()


def open_browser_later(url):
    timer = threading.Timer(1.2, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main():
    args = parse_args()
    local_url = f"http://localhost:{args.port}"

    print("Medical Agent Web Launcher")
    print("=" * 28)
    print(f"Local URL: {local_url}")
    print("If this is your first run, the web page will guide you through API setup.")
    print("Press Ctrl+C to stop the server.")

    if not args.no_browser:
        open_browser_later(local_url)

    env = os.environ.copy()
    env["MEDICAL_AGENT_HOST"] = args.host
    env["MEDICAL_AGENT_PORT"] = str(args.port)
    subprocess.run([sys.executable, "app.py"], env=env)


if __name__ == "__main__":
    main()
