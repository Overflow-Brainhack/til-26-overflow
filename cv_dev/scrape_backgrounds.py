"""
Download background images from Mapillary API v4 (no cap — collects everything available).

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
# Skewed toward natural/open scenes (forest, coast, farmland, desert, sea) to
# avoid backgrounds cluttered with vehicles/people. ~10 urban centres kept for
# coverage. Each centre is expanded into 5 non-overlapping 0.01°×0.01° boxes.
_CENTERS = [
    # --- Forests & woodland ---
    (25.15, 65.10, "Finnish boreal forest"),
    (27.80, 63.40, "Finnish lakeland"),
    (15.50, 59.30, "Swedish forest"),
    (10.40, 60.50, "Norwegian forest"),
    (7.50, 47.60, "Black Forest / Jura"),
    (-1.50, 52.10, "English countryside"),
    (-4.50, 54.30, "Lake District UK"),
    (-77.40, 44.20, "Algonquin forest Canada"),
    (-85.50, 46.50, "Upper Michigan forest"),
    (-122.80, 48.00, "Olympic Peninsula rainforest"),
    (-123.50, 48.70, "Vancouver Island forest"),
    (135.50, -37.00, "Victorian mountain ash forest"),
    (172.65, -43.55, "New Zealand Southern Alps forest"),
    (103.80, 1.45, "Singapore / Borneo rainforest edge"),
    (114.80, 4.90, "Brunei rainforest"),
    # --- Coastlines & sea cliffs ---
    (-9.40, 39.50, "Portuguese Atlantic coast"),
    (-8.60, 42.10, "Galician coast Spain"),
    (-5.50, 36.10, "Strait of Gibraltar coast"),
    (-63.90, 45.10, "Nova Scotia rocky coast"),
    (-70.20, 41.70, "Cape Cod coast"),
    (-124.00, 41.00, "Northern California coast"),
    (-156.50, 20.80, "Hawaii coastline"),
    (3.00, 43.30, "Languedoc coast France"),
    (14.50, 42.60, "Adriatic coast Croatia"),
    (24.00, 37.80, "Greek island coast"),
    (28.20, 36.60, "Turkish Aegean coast"),
    (172.70, -43.80, "New Zealand Kaikoura coast"),
    (151.30, -33.70, "Sydney northern beaches"),
    (115.20, -33.80, "Western Australia coast"),
    (36.80, -1.20, "Kenya coast"),
    # --- Farmland & open countryside ---
    (0.50, 51.80, "English wheat fields"),
    (3.50, 50.80, "Belgian/French farmland"),
    (9.00, 53.50, "North German plains"),
    (17.00, 48.50, "Slovak/Hungarian farmland"),
    (-96.00, 41.50, "Nebraska plains"),
    (-100.50, 37.00, "Kansas wheat plains"),
    (-1.00, 46.50, "French bocage countryside"),
    (11.50, 44.50, "Po Valley Italy"),
    (138.60, -34.50, "South Australian farmland"),
    (-64.50, -31.50, "Argentine pampas"),
    # --- Desert & arid ---
    (-111.90, 33.50, "Arizona Sonoran desert"),
    (-117.00, 35.50, "Mojave desert"),
    (55.00, 24.00, "UAE / Oman desert"),
    (45.00, 22.00, "Empty Quarter Arabia"),
    (9.50, 30.50, "Tunisian desert"),
    (25.00, 27.00, "Egyptian Western Desert"),
    (136.00, -25.00, "Australian outback"),
    (-68.50, -24.50, "Atacama desert Chile"),
    # --- Mountains & highlands ---
    (7.70, 46.00, "Swiss Alps"),
    (10.90, 47.40, "Austrian Tyrolean Alps"),
    (14.00, 46.30, "Slovenian Julian Alps"),
    (-105.60, 40.40, "Colorado Rockies"),
    (-119.50, 37.80, "Sierra Nevada California"),
    (-113.50, 51.20, "Canadian Rockies"),
    (-70.60, -33.40, "Andes Chile/Argentina"),
    (85.00, 28.00, "Nepal Himalayan foothills"),
    (80.00, 31.00, "Indian Himachal Pradesh"),
    (44.50, 42.50, "Caucasus mountains Georgia"),
    # --- Savannah & grassland ---
    (36.80, -2.60, "Kenyan savannah"),
    (31.50, -2.80, "Rwandan highlands"),
    (27.00, -17.00, "Zambian bush"),
    (-47.00, -15.50, "Brazilian cerrado"),
    (144.00, -23.50, "Queensland outback"),
    # --- Lakes & rivers (open water backgrounds) ---
    (25.60, 61.80, "Finnish lake district"),
    (-78.90, 44.00, "Lake Ontario shore"),
    (-86.50, 44.50, "Lake Michigan shore"),
    (6.50, 46.50, "Lake Geneva shore"),
    (11.30, 46.10, "Lake Garda Italy"),
    (36.50, 0.50, "Lake Nakuru Kenya"),
    # --- A handful of urban centres for coverage ---
    (-73.80, 40.70, "New York outer borough"),
    (-122.40, 37.80, "San Francisco"),
    (2.35, 48.90, "Paris suburbs"),
    (13.45, 52.50, "Berlin"),
    (139.75, 35.70, "Tokyo suburbs"),
    (127.05, 37.60, "Seoul"),
    (28.05, -26.20, "Johannesburg"),
    (36.85, -1.30, "Nairobi city"),
    (151.20, -33.85, "Sydney"),
    (-58.40, -34.60, "Buenos Aires"),
]


def _zones(lon: float, lat: float) -> list[tuple[float, float, float, float]]:
    """Five non-overlapping 0.01°×0.01° boxes centred on (lon, lat)."""
    h, s = 0.005, 0.015
    return [
        (lon - h, lat - h, lon + h, lat + h),  # centre
        (lon - h, lat + s - h, lon + h, lat + s + h),  # north
        (lon - h, lat - s - h, lon + h, lat - s + h),  # south
        (lon + s - h, lat - h, lon + s + h, lat + h),  # east
        (lon - s - h, lat - h, lon - s + h, lat + h),  # west
    ]


# 76 regions × 5 zones = 380 bboxes
BBOXES = [box for lon, lat, *_ in _CENTERS for box in _zones(lon, lat)]


def fetch_images(bbox: tuple[float, ...], token: str, limit: int) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "bbox": bbox_str,
        "fields": "id,thumb_1024_url",
        "limit": limit,
        "is_pano": "true",  # panoramic shots → open scenes, fewer vehicles/people
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


def scrape_backgrounds(token: str) -> None:
    BACKGROUNDS_PATH.mkdir(parents=True, exist_ok=True)

    existing = set(p.stem for p in BACKGROUNDS_PATH.glob("*.jpg"))
    print(
        f"Already have {len(existing)} backgrounds — scraping all {len(BBOXES)} bboxes."
    )

    collected = len(existing)

    for bbox in tqdm(BBOXES, "Fetching from regions", colour="Blue"):
        try:
            images = fetch_images(bbox, token, limit=5)
        except Exception as e:
            print(f"  API error for bbox {bbox}: {e}")
            time.sleep(2)
            continue

        for img in images:
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
    args = parser.parse_args()

    if not args.token:
        print(
            "Error: Mapillary token required.\n"
            "  Set MAPILLARY_TOKEN env var or pass --token <token>\n"
            "  Get a free token at https://www.mapillary.com/dashboard/developers"
        )
        raise SystemExit(1)

    scrape_backgrounds(args.token)
