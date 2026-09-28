"""Download player scouting datasets.

Primary sources:
1. Curated FBref Big 5 advanced season player stats (standard, shooting, passing, possession, defense, misc)
   via worldfootballR_data release assets.
2. Fantasy Premier League (FPL) historical player seasons (2017-18 to 2023-24)
   via vaastav/Fantasy-Premier-League.

Output Directory: data/raw/scouting/
"""

from pathlib import Path
import sys
import pyreadr
import requests

FBREF_BASE_URL = "https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats"
FBREF_TABLES = [
    "big5_player_standard",
    "big5_player_shooting",
    "big5_player_passing",
    "big5_player_possession",
    "big5_player_defense",
    "big5_player_misc",
]

FPL_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FPL_SEASONS = [
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

DEFAULT_OUTPUT_DIR = Path("data/raw/scouting")


def download_fbref_table(
    table_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Download an FBref advanced player statistics table (.rds) and export to CSV.

    Args:
        table_name: Base name (e.g. 'big5_player_standard')
        output_dir: Destination directory
        force: Redownload if True

    Returns:
        Path to exported CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rds_path = output_dir / f"{table_name}.rds"
    csv_path = output_dir / f"{table_name}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1024 and not force:
        print(f"[{table_name}] Scouting CSV already exists ({csv_path.stat().st_size:,} bytes). Skipping.")
        return csv_path

    url = f"{FBREF_BASE_URL}/{table_name}.rds"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PremAI/1.0"}

    print(f"[{table_name}] Downloading FBref table from {url}...")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=90) as r:
            r.raise_for_status()
            with open(rds_path, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f_out.write(chunk)

        print(f"[{table_name}] Downloaded {rds_path.stat().st_size:,} bytes. Parsing and exporting to CSV...")
        result = pyreadr.read_r(str(rds_path))
        df = result[None]
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[{table_name}] Successfully exported: {csv_path} ({df.shape[0]:,} rows, {df.shape[1]} cols)")
        return csv_path

    except Exception as exc:
        print(f"[{table_name}] ERROR downloading/parsing FBref table: {exc}", file=sys.stderr)
        raise


def download_fpl_season(
    season: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Download seasonal cleaned player metrics from vaastav/Fantasy-Premier-League."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fpl_{season}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1024 and not force:
        print(f"[FPL {season}] Already exists ({csv_path.stat().st_size:,} bytes). Skipping.")
        return csv_path

    url = f"{FPL_BASE_URL}/{season}/cleaned_players.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PremAI/1.0"}

    print(f"[FPL {season}] Downloading from {url}...")
    try:
        res = requests.get(url, headers=headers, timeout=20)
        res.raise_for_status()
        csv_path.write_bytes(res.content)
        print(f"[FPL {season}] Successfully saved: {csv_path} ({len(res.content):,} bytes)")
        return csv_path
    except Exception as exc:
        print(f"[FPL {season}] ERROR downloading FPL data: {exc}", file=sys.stderr)
        raise


def download_all_scouting_data(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download and process all scouting data sources."""
    results = {}
    print(f"=== Starting Scouting Data Ingestion ({len(FBREF_TABLES)} FBref tables + {len(FPL_SEASONS)} FPL seasons) ===")
    for table_name in FBREF_TABLES:
        results[table_name] = download_fbref_table(
            table_name=table_name,
            output_dir=output_dir,
            force=force,
        )

    for season in FPL_SEASONS:
        key = f"fpl_{season}"
        results[key] = download_fpl_season(
            season=season,
            output_dir=output_dir,
            force=force,
        )
    print("=== Scouting Data Ingestion Complete ===\n")
    return results


if __name__ == "__main__":
    download_all_scouting_data()
