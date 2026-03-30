#!/usr/bin/env python3
"""Download and convert JMdict to SQLite for local development.

Downloads from scriptin/jmdict-simplified GitHub releases,
converts JSON to SQLite using jmdict_to_sqlite.py.

Usage: python scripts/download_jmdict.py
"""

import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

JMDICT_VERSION = "3.5.0"
RELEASE_URL = (
    f"https://github.com/scriptin/jmdict-simplified/releases/download/"
    f"{JMDICT_VERSION}/jmdict-eng-{JMDICT_VERSION}.json.tgz"
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "static"
OUTPUT_PATH = OUTPUT_DIR / "jmdict.db"


def main():
    if OUTPUT_PATH.exists():
        print(f"JMdict already exists at {OUTPUT_PATH}")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tgz_path = os.path.join(tmpdir, "jmdict.tgz")
        print(f"Downloading {RELEASE_URL}...")
        urllib.request.urlretrieve(RELEASE_URL, tgz_path)

        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        json_path = os.path.join(tmpdir, f"jmdict-eng-{JMDICT_VERSION}.json")
        if not os.path.exists(json_path):
            # Find the JSON file
            for f in os.listdir(tmpdir):
                if f.endswith(".json"):
                    json_path = os.path.join(tmpdir, f)
                    break

        print(f"Converting to SQLite at {OUTPUT_PATH}...")
        # Import the converter
        sys.path.insert(0, str(ROOT / "scripts"))
        from jmdict_to_sqlite import convert
        convert(json_path, str(OUTPUT_PATH))

    print("Done.")


if __name__ == "__main__":
    main()
