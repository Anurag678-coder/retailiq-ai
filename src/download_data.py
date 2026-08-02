"""
download_data.py
Downloads the real "Online Retail II" dataset (UCI Machine Learning Repository,
donated by Dr. Daqing Chen) and saves it as a single CSV at:

    data/raw/online_retail.csv

This is a real transactional dataset for a UK-based online gift retailer
(01/12/2009 - 09/12/2011). No synthetic data is generated anywhere in this
project's default pipeline.

Run:
    python src/download_data.py

If you're offline or the automatic download fails (school/office networks
often block this), download manually instead:

    1. UCI:    https://archive.ics.uci.edu/dataset/502/online+retail+ii
    2. Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Save the file as CSV at: data/raw/online_retail.csv
(If you only have the .xlsx, open it and "Save As" CSV, or run:
    python -c "import pandas as pd; pd.read_excel('online_retail_II.xlsx', sheet_name=None)"
 and concatenate both sheets before saving.)
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RAW_DIR / "online_retail.csv"

UCI_XLSX_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)


def download_via_ucimlrepo() -> pd.DataFrame:
    """Preferred path: the official `ucimlrepo` package (pip install ucimlrepo)."""
    from ucimlrepo import fetch_ucirepo

    print("Fetching dataset via ucimlrepo (id=502, Online Retail II)...")
    dataset = fetch_ucirepo(id=502)
    df = dataset.data.features.copy()
    if dataset.data.targets is not None and len(dataset.data.targets.columns) > 0:
        df = pd.concat([df, dataset.data.targets], axis=1)
    return df


def download_via_requests() -> pd.DataFrame:
    """Fallback: download the zipped Excel workbook directly and combine both sheets."""
    import io
    import zipfile
    import requests

    print("Fetching dataset zip directly from archive.ics.uci.edu...")
    resp = requests.get(UCI_XLSX_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(xlsx_name) as f:
            sheets = pd.read_excel(f, sheet_name=None)

    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def main():
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists — delete it first if you want to re-download.")
        return

    df = None
    for fn in (download_via_ucimlrepo, download_via_requests):
        try:
            df = fn()
            break
        except Exception as e:
            print(f"  {fn.__name__} failed: {e}")

    if df is None:
        print(
            "\nAutomatic download failed (likely no internet access, or the network "
            "blocks archive.ics.uci.edu). Please download the dataset manually:\n\n"
            "  UCI:    https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            "  Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n\n"
            f"Then save it as CSV at: {OUTPUT_PATH}"
        )
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
