#!/usr/bin/env python3
"""Download public-domain CLIP fixture photos (run-once).

NASA images are public domain. Each entry below is (filename, source_url).
Verify each URL resolves and is PD before committing the output.
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "clip"
MAX_SIDE = 512

# Diverse public-domain content (NASA image library). Confirm each resolves.
SOURCES: list[tuple[str, str]] = [
    ("earth_blue_marble.jpg", "https://images-assets.nasa.gov/image/PIA18033/PIA18033~orig.jpg"),
    ("apollo17_moon.jpg", "https://images-assets.nasa.gov/image/as17-148-22727/as17-148-22727~orig.jpg"),
    ("nebula_hubble.jpg", "https://images-assets.nasa.gov/image/GSFC_20171208_Archive_e000075/GSFC_20171208_Archive_e000075~orig.jpg"),
    ("astronaut_spacewalk.jpg", "https://images-assets.nasa.gov/image/iss040e090540/iss040e090540~orig.jpg"),
    ("mars_curiosity.jpg", "https://images-assets.nasa.gov/image/PIA16239/PIA16239~orig.jpg"),
    ("nasa_pia12348.jpg", "https://images-assets.nasa.gov/image/PIA12348/PIA12348~orig.jpg"),
    ("nasa_pia17011.jpg", "https://images-assets.nasa.gov/image/PIA17011/PIA17011~orig.jpg"),
    ("nasa_pia03883.jpg", "https://images-assets.nasa.gov/image/PIA03883/PIA03883~orig.jpg"),
]


def fetch_one(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "parity-fixtures/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in SOURCES:
        print(f"[fetch] {name} <- {url}")
        img = Image.open(io.BytesIO(fetch_one(url))).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(OUT / name, format="JPEG", quality=90)
        rows.append((name, url))
    print(f"\n[done] wrote {len(rows)} images to {OUT}")
    print("\nProvenance rows (paste into tests/fixtures/README.md):")
    for name, url in rows:
        print(f"| {name} | {url} | NASA — public domain |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
