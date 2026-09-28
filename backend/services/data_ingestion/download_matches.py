"""Download historical Premier League match datasets from football-data.co.uk.

Target Seasons: 2017/18 through 2023/24
Output structure: data/raw/matches/<season>/E0.csv
"""

from pathlib import Path
import sys
import requests

# Base URL and season mapping for football-data.co.uk
BASE_URL = "https://football-data.co.uk/mmz4281"
SEASONS = {
    "2017-18": "1718",
    "2018-19": "1819",
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
}

DEFAULT_OUTPUT_DIR = Path("data/raw/matches")


def download_match_season(
    season_label: str,
    season_code: str,
    output_base_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Download E0.csv for a specific Premier League season.

    Args:
        season_label: Format 'YYYY-YY' (e.g. '2023-24')
        season_code: Format 'YYZZ' (e.g. '2324')
        output_base_dir: Directory under which <season>/E0.csv will be saved.
        force: If True, redownload even if file exists.

    Returns:
        Path to the downloaded E0.csv file.
    """
    season_dir = output_base_dir / season_label
    season_dir.mkdir(parents=True, exist_ok=True)
    target_file = season_dir / "E0.csv"

    if target_file.exists() and target_file.stat().st_size > 1024 and not force:
        print(f"[{season_label}] Match data already exists ({target_file.stat().st_size:,} bytes). Skipping.")
        return target_file

    url = f"{BASE_URL}/{season_code}/E0.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PremAI/1.0"}

    print(f"[{season_label}] Downloading from {url}...")
    try:
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=20)
        response.raise_for_status()

        # Sanity check: must contain CSV header
        if b"HomeTeam" not in response.content and b"Date" not in response.content:
            raise ValueError(f"Downloaded content from {url} does not appear to be valid match CSV data.")

        target_file.write_bytes(response.content)
        print(f"[{season_label}] Successfully saved: {target_file} ({len(response.content):,} bytes)")
        return target_file

    except Exception as exc:
        print(f"[{season_label}] ERROR downloading match data: {exc}", file=sys.stderr)
        raise


def download_all_matches(
    output_base_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download Premier League match CSVs for all target seasons."""
    results = {}
    print(f"=== Starting Match Data Ingestion ({len(SEASONS)} seasons) ===")
    for season_label, season_code in SEASONS.items():
        results[season_label] = download_match_season(
            season_label=season_label,
            season_code=season_code,
            output_base_dir=output_base_dir,
            force=force,
        )
    print("=== Match Data Ingestion Complete ===\n")
    return results


if __name__ == "__main__":
    download_all_matches()
