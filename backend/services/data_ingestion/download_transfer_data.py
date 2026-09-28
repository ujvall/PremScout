"""Download Transfermarkt dataset tables from dcaribou/transfermarkt-datasets.

Source: Cloudflare R2 open bucket hosted by dcaribou
Base URL: https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data
Output Directory: data/raw/transfermarkt/
"""

import gzip
from pathlib import Path
import shutil
import sys
import requests

R2_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

TABLES = [
    "players",
    "player_valuations",
    "transfers",
    "appearances",
    "clubs",
    "games",
]

DEFAULT_OUTPUT_DIR = Path("data/raw/transfermarkt")


def download_transfer_table(
    table_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Download and decompress a specific Transfermarkt table.

    Args:
        table_name: Name of table (e.g. 'players', 'player_valuations')
        output_dir: Destination folder
        force: If True, redownload even if target exists

    Returns:
        Path to decompressed .csv file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gz_path = output_dir / f"{table_name}.csv.gz"
    csv_path = output_dir / f"{table_name}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1024 and not force:
        print(f"[{table_name}] CSV already exists ({csv_path.stat().st_size:,} bytes). Skipping.")
        return csv_path

    url = f"{R2_BASE_URL}/{table_name}.csv.gz"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PremAI/1.0"}

    print(f"[{table_name}] Downloading {url}...")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f_out.write(chunk)

        print(f"[{table_name}] Downloaded {gz_path.stat().st_size:,} bytes. Decompressing to {csv_path.name}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"[{table_name}] Successfully prepared: {csv_path} ({csv_path.stat().st_size:,} bytes)")
        return csv_path

    except Exception as exc:
        print(f"[{table_name}] ERROR downloading/extracting table: {exc}", file=sys.stderr)
        if csv_path.exists():
            csv_path.unlink()
        raise


def download_all_transfer_data(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download and extract all required Transfermarkt tables."""
    results = {}
    print(f"=== Starting Transfermarkt Ingestion ({len(TABLES)} tables) ===")
    for table_name in TABLES:
        results[table_name] = download_transfer_table(
            table_name=table_name,
            output_dir=output_dir,
            force=force,
        )
    print("=== Transfermarkt Ingestion Complete ===\n")
    return results


if __name__ == "__main__":
    download_all_transfer_data()
