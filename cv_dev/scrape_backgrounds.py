"""
Download 500 ground-level background images from Mapillary API v4.

Requires a free Mapillary access token:
  1. Sign up at https://www.mapillary.com/
  2. Go to https://www.mapillary.com/dashboard/developers → create an application
  3. Copy the Client Token
  4. Set env variable: export MAPILLARY_TOKEN=<your_token>
     or pass --token <token> on the command line

Images are saved to cv_dev/backgrounds/ as JPEG.
"""

from cv_dev.consts import BACKGROUNDS_PATH

import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Region centres (lon, lat, label).
# Each is expanded into 5 non-overlapping 0.01°×0.01° boxes (API maximum).
_CENTERS = [
    # Southeast Asia
    (103.85,  1.35,  "Singapore"),
    (100.65,  13.80, "Bangkok"),
    (106.80,  10.85, "Ho Chi Minh City"),
    (104.05,  3.30,  "Kuala Lumpur"),
    (114.25,  22.35, "Hong Kong"),
    (121.05,  14.65, "Manila"),
    (106.95,  -6.20, "Jakarta"),
    # Europe
    (2.35,    48.90, "Paris"),
    (13.45,   52.50, "Berlin"),
    (-0.05,   51.50, "London"),
    (4.90,    52.40, "Amsterdam"),
    (18.15,   59.40, "Stockholm"),
    (25.05,   60.20, "Helsinki"),
    (-1.50,   52.10, "English countryside"),
    (7.50,    47.60, "Swiss/German forest"),
    (16.35,   48.20, "Vienna"),
    (2.15,    41.40, "Barcelona"),
    (12.50,   41.90, "Rome"),
    # North America
    (-73.80,  40.70, "New York"),
    (-87.60,  41.90, "Chicago"),
    (-122.40, 37.80, "San Francisco"),
    (-79.40,  43.70, "Toronto"),
    (-63.90,  45.10, "Nova Scotia"),
    (-118.25, 34.05, "Los Angeles"),
    (-80.20,  25.80, "Miami"),
    (-122.35, 47.65, "Seattle"),
    # Nordic / Northern Europe
    (25.15,   65.10, "Finnish forest"),
    (10.75,   59.90, "Oslo"),
    # Middle East
    (55.35,   25.20, "Dubai"),
    (46.75,   24.70, "Riyadh"),
    (35.20,   31.80, "Jerusalem"),
    # East Asia
    (139.75,  35.70, "Tokyo"),
    (127.05,  37.60, "Seoul"),
    (121.55,  31.20, "Shanghai"),
    (104.05,  30.65, "Chengdu"),
    # South Asia
    (72.85,   19.10, "Mumbai"),
    (77.20,   28.60, "Delhi"),
    (88.35,   22.55, "Kolkata"),
    (80.25,   13.10, "Chennai"),
    # Africa
    (28.05,   -26.20, "Johannesburg"),
    (36.85,   -1.30,  "Nairobi"),
    (31.25,   30.05,  "Cairo"),
    (3.40,    6.45,   "Lagos"),
    (32.55,   0.30,   "Kampala"),
    # South America
    (-43.20,  -22.90, "Rio de Janeiro"),
    (-46.65,  -23.55, "São Paulo"),
    (-58.40,  -34.60, "Buenos Aires"),
    (-74.10,  4.70,   "Bogotá"),
    # Oceania
    (151.20,  -33.85, "Sydney"),
    (144.95,  -37.80, "Melbourne"),
    (172.65,  -43.55, "Christchurch"),
]


def _zones(lon: float, lat: float) -> list[tuple[float, float, float, float]]:
    """Five non-overlapping 0.01°×0.01° boxes centred on (lon, lat)."""
    h, s = 0.005, 0.015
    return [
        (lon - h, lat - h, lon + h, lat + h),           # centre
        (lon - h, lat + s - h, lon + h, lat + s + h),   # north
        (lon - h, lat - s - h, lon + h, lat - s + h),   # south
        (lon + s - h, lat - h, lon + s + h, lat + h),   # east
        (lon - s - h, lat - h, lon - s + h, lat + h),   # west
    ]


# 51 regions × 5 zones = 255 bboxes
BBOXES = [box for lon, lat, *_ in _CENTERS for box in _zones(lon, lat)]

PER_BBOX = 20  # target driven by scrape_backgrounds(); this is the per-call cap


def fetch_images(bbox: tuple[float, ...], token: str, limit: int) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "bbox": bbox_str,
        "fields": "id,thumb_1024_url",
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def download_image(url: str, dest: Path, token: str) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  Failed to download {dest.name}: {e}")
        return False


def scrape_backgrounds(token: str, target: int = 500) -> None:
    BACKGROUNDS_PATH.mkdir(parents=True, exist_ok=True)

    existing = set(p.stem for p in BACKGROUNDS_PATH.glob("*.jpg"))
    print(f"Already have {len(existing)} backgrounds, targeting {target} total.")

    collected = len(existing)
    per_bbox = max(1, (target - collected + len(BBOXES) - 1) // len(BBOXES))

    for bbox in tqdm(BBOXES, "Fetching from regions", colour="Purple"):
        if collected >= target:
            break

        try:
            images = fetch_images(bbox, token, limit=min(per_bbox + 5, 50))
        except Exception as e:
            print(f"  API error for bbox {bbox}: {e}")
            time.sleep(2)
            continue

        for img in images:
            if collected >= target:
                break
            img_id = str(img.get("id", ""))
            if img_id in existing:
                continue
            thumb_url = img.get("thumb_1024_url")
            if not thumb_url:
                continue
            dest = BACKGROUNDS_PATH / f"{img_id}.jpg"
            if download_image(thumb_url, dest, token):
                existing.add(img_id)
                collected += 1

        time.sleep(0.3)  # polite rate limit

    print(f"\nDone. {collected} backgrounds saved to {BACKGROUNDS_PATH}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN", ""))
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()

    if not args.token:
        print(
            "Error: Mapillary token required.\n"
            "  Set MAPILLARY_TOKEN env var or pass --token <token>\n"
            "  Get a free token at https://www.mapillary.com/dashboard/developers"
        )
        raise SystemExit(1)

    scrape_backgrounds(args.token, args.target)
