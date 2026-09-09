"""
Download and unpack the pre-processed EMBER2024 WIN32 dataset.

Downloads a ~8.5 GB zip from Google Drive containing four parquet files
(win32_detection_train_20pct.parquet, win32_test_detection.parquet,
win32_behavior_train_20pct.parquet, win32_test_behavior.parquet) and
extracts them into win32_data/ at the repo root.

Needs ~20-30 GB of free disk space (zip + extracted parquet files).

If this fails (e.g. Google Drive quota/link issues), fall back to
notebooks/hf-download-win32.ipynb, which rebuilds the same data from the
~74 GB raw EMBER2024 dataset on Hugging Face (needs 80-100 GB free).

Usage:
    python backend/download_data.py
"""

import shutil
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "win32_data"
ZIP_PATH = ROOT / "win32_data.zip"
FILE_ID = "1UwYNfKhweCbJHHMCwY9FO5ha77FcZUGo"

EXPECTED_FILES = [
    "win32_detection_train_20pct.parquet",
    "win32_test_detection.parquet",
    "win32_behavior_train_20pct.parquet",
    "win32_test_behavior.parquet",
]


def already_downloaded() -> bool:
    return all((DATA_DIR / name).exists() for name in EXPECTED_FILES)


def download():
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading dataset zip (file id {FILE_ID}) to {ZIP_PATH} ...")
        gdown.download(id=FILE_ID, output=str(ZIP_PATH), quiet=False)
    else:
        print(f"Zip already present at {ZIP_PATH}, skipping download.")

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        raise RuntimeError(
            "Download failed or produced an empty file. This is often a Google "
            "Drive quota/permission issue for large files. Fall back to "
            "notebooks/hf-download-win32.ipynb instead."
        )

    print("Downloaded file size:", ZIP_PATH.stat().st_size, "bytes")

    print(f"Extracting into {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    # If extraction created a nested data/ folder, move parquet files up
    nested_data = DATA_DIR / "data"
    if nested_data.exists():
        for parquet_file in nested_data.glob("*.parquet"):
            shutil.move(str(parquet_file), str(DATA_DIR / parquet_file.name))
        shutil.rmtree(nested_data)

    macos_folder = DATA_DIR / "__MACOSX"
    if macos_folder.exists():
        shutil.rmtree(macos_folder)

    ZIP_PATH.unlink()

    print("Dataset ready:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(" -", f.name)


def main():
    if already_downloaded():
        print(f"All expected parquet files already present in {DATA_DIR}, skipping download.")
        return

    download()

    missing = [name for name in EXPECTED_FILES if not (DATA_DIR / name).exists()]
    if missing:
        raise RuntimeError(
            f"Extraction finished but these expected files are missing: {missing}. "
            "The zip contents may not match what this script expects."
        )


if __name__ == "__main__":
    main()
