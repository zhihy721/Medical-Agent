#!/usr/bin/env python3

import os
import subprocess
import sys


def main():
    print("Medical Agent Demo Launcher")
    print("=" * 28)

    if not os.path.exists("config.env"):
        print("config.env not found.")
        print("Please copy config.env.example to config.env and fill in your DeepSeek key.")
        return

    print("Starting Flask app at http://localhost:5000")
    subprocess.run([sys.executable, "app.py"])


if __name__ == "__main__":
    main()
