"""
Download natural-scene background images from the Pexels API.
Searches landscape-oriented nature terms only — no cities, roads, or vehicles.

Requires a free Pexels API key:
  1. Sign up at https://www.pexels.com/api/
  2. Set env variable: export PEXELS_TOKEN=<your_token>
     or pass --token <token> on the command line

Images are saved to cv_dev/backgrounds/ as JPEG.
Rate limits: 200 req/hour, 20 000 req/month (free tier).
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

# Each query fetches up to PER_PAGE landscape photos from Pexels.
# Total budget: len(QUERIES) * PER_PAGE images maximum.
QUERIES = [
    "forest",
    "jungle rainforest",
    "woodland trees",
    "desert sand dunes",
    "arid landscape",
    "beach coastline",
    "ocean waves sea",
    "rocky cliff coast",
    "grassland meadow",
    "savanna",
    "prairie plains",
    "mountain landscape",
    "alpine highland",
    "lake nature",
    "river valley",
    "wetland marsh",
    "farmland countryside",
    "rural field",
    "snow landscape",
    "tundra arctic",
    "sky clouds",
    "aerial landscape",
    "rocky terrain",
    "volcanic landscape",
    "tropical island",
]

PER_PAGE = 80  # Pexels API maximum per request


def fetch_page(query: str, token: str, page: int) -> list[dict]:
    resp = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": token},
        params={
            "query": query,
            "per_page": PER_PAGE,
            "page": page,
            "orientation": "landscape",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("photos", [])


def download_image(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  Failed: {dest.name}: {e}")
        return False


def scrape_backgrounds(token: str) -> None:
    BACKGROUNDS_PATH.mkdir(parents=True, exist_ok=True)
    existing = set(p.stem for p in BACKGROUNDS_PATH.glob("*.jpg"))
    print(
        f"Already have {len(existing)} backgrounds — fetching up to {len(QUERIES) * PER_PAGE} more."
    )

    collected = len(existing)

    for query in tqdm(QUERIES, "Queries", colour="Blue"):
        try:
            photos = fetch_page(query, token, page=1)
        except Exception as e:
            print(f"  API error for '{query}': {e}")
            time.sleep(2)
            continue

        for photo in photos:
            photo_id = str(photo["id"])
            if photo_id in existing:
                continue
            url = photo["src"].get("large2x") or photo["src"].get("large")
            if not url:
                continue
            dest = BACKGROUNDS_PATH / f"{photo_id}.jpg"
            if download_image(url, dest):
                existing.add(photo_id)
                collected += 1

        time.sleep(0.4)  # stay well within 200 req/hour

    print(f"\nDone. {collected} backgrounds saved to {BACKGROUNDS_PATH}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("PEXELS_TOKEN", ""))
    args = parser.parse_args()

    if not args.token:
        print(
            "Error: Pexels API key required.\n"
            "  Set PEXELS_TOKEN env var or pass --token <token>\n"
            "  Get a free key at https://www.pexels.com/api/"
        )
        raise SystemExit(1)

    scrape_backgrounds(args.token)
