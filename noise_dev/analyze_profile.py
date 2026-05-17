#!/usr/bin/env python3
"""Analyze noiser.profile and report the slowest functions."""

import pstats
import sys
from pathlib import Path

PROFILE = Path(__file__).parent / "noiser.profile"
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OWN_CODE = str(Path(__file__).parent / "src")


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


stats = pstats.Stats(str(PROFILE))
stats.strip_dirs()

# ── Top N by cumulative time (wall time including callees) ──────────────────
print_section(f"Top {TOP_N} by CUMULATIVE time")
stats.sort_stats("cumulative")
stats.print_stats(TOP_N)

# ── Top N by total self time (time spent inside the function itself) ─────────
print_section(f"Top {TOP_N} by SELF (tottime) time")
stats.sort_stats("tottime")
stats.print_stats(TOP_N)

# ── Only functions from noise/src/ ──────────────────────────────────────────
print_section("OWN CODE (noise_manager.py) — by cumulative time")
stats.sort_stats("cumulative")
stats.print_stats("noise_manager")
