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

import argparse
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

BACKGROUNDS_PATH = Path("cv_dev/backgrounds")

# Diverse bboxes: (min_lon, min_lat, max_lon, max_lat)
# ~25 images each → ~500 total
BBOXES = [
    # Southeast Asia (most likely matches test distribution)
    (103.6, 1.2, 104.1, 1.5),    # Singapore
    (100.4, 13.6, 100.9, 14.0),  # Bangkok outskirts
    (106.6, 10.7, 107.0, 11.0),  # Ho Chi Minh City
    (103.8, 3.1, 104.3, 3.5),    # Malaysia
    (114.1, 22.2, 114.4, 22.5),  # Hong Kong
    (120.9, 14.5, 121.2, 14.8),  # Manila
    (106.8, -6.3, 107.1, -6.1),  # Jakarta
    # Europe — varied terrain
    (2.2, 48.8, 2.5, 49.0),      # Paris suburbs
    (13.3, 52.4, 13.6, 52.6),    # Berlin
    (-0.2, 51.4, 0.1, 51.6),     # London
    (4.8, 52.3, 5.0, 52.5),      # Amsterdam
    (18.0, 59.3, 18.3, 59.5),    # Stockholm
    (24.9, 60.1, 25.2, 60.3),    # Helsinki
    # North America
    (-73.9, 40.6, -73.7, 40.8),  # New York outer borough
    (-87.7, 41.8, -87.5, 42.0),  # Chicago
    (-122.5, 37.7, -122.3, 37.9),# San Francisco
    (-79.5, 43.6, -79.3, 43.8),  # Toronto
    # Rural / natural (forests, fields)
    (-1.6, 52.0, -1.4, 52.2),    # English countryside
    (7.4, 47.5, 7.6, 47.7),      # Swiss/German forest
    (-64.0, 45.0, -63.8, 45.2),  # Nova Scotia coast
    (25.0, 65.0, 25.3, 65.2),    # Finnish forest
    # Middle East / arid
    (55.2, 25.1, 55.5, 25.3),    # Dubai
    (46.6, 24.6, 46.9, 24.8),    # Riyadh
    # East Asia
    (139.6, 35.6, 139.9, 35.8),  # Tokyo suburbs
    (126.9, 37.5, 127.2, 37.7),  # Seoul
    (121.4, 31.1, 121.7, 31.3),  # Shanghai suburbs
]

PER_BBOX = 20  # ~500 total (26 bboxes × 20 = 520)


def fetch_images(bbox: tuple, token: str, limit: int) -> list[dict]:
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

    for bbox in tqdm(BBOXES, "Fetching from regions"):
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
