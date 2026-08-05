"""
Dataset downloader for AccidentAI training pipeline.

Downloads from:
  1. HuggingFace (free, no key needed):
     - M-ArslanArshad/Car_Accidents_and_deformation_dataset
     - tiya1012/vit-accident-image (model weights, not dataset)

  2. Kaggle (requires ~/.kaggle/kaggle.json):
     - arnav3107/accident-detection-dataset
     - sartajbhuvaji/car-accident-dataset
     - khotijaoks/car-damages-detection

Usage:
    python training/download_datasets.py
    python training/download_datasets.py --hf-only      # skip Kaggle
    python training/download_datasets.py --list         # show available datasets
"""

import os
import json
import shutil
import argparse
from pathlib import Path


# ─── Directory layout ────────────────────────────────────────────────────────────

DATA_ROOT = Path("training/data")
IMG_DIR   = DATA_ROOT / "images"
VID_DIR   = DATA_ROOT / "videos"
VIDEO_ACCIDENT = VID_DIR / "accident"
VIDEO_NO_ACCIDENT = VID_DIR / "no_accident"

ACCIDENT_TRAIN    = IMG_DIR / "accident" / "train"
ACCIDENT_VAL      = IMG_DIR / "accident" / "val"
NO_ACCIDENT_TRAIN = IMG_DIR / "no_accident" / "train"
NO_ACCIDENT_VAL   = IMG_DIR / "no_accident" / "val"


def setup_dirs():
    for d in [ACCIDENT_TRAIN, ACCIDENT_VAL, NO_ACCIDENT_TRAIN, NO_ACCIDENT_VAL, VIDEO_ACCIDENT, VIDEO_NO_ACCIDENT]:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created.")


# ─── HuggingFace datasets (no auth required) ─────────────────────────────────────

def download_hf_car_accident_deformation():
    """
    Download M-ArslanArshad/Car_Accidents_and_deformation_dataset from HuggingFace.
    ~4K images across: normal, collision, deformed_vehicle classes.
    """
    print("\n[HuggingFace] Downloading Car_Accidents_and_deformation_dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("M-ArslanArshad/Car_Accidents_and_deformation_dataset")
        print(f"  Dataset loaded: {ds}")

        saved_acc = 0
        saved_no  = 0

        for split_name, split in ds.items():
            is_train = split_name == "train"
            acc_dir  = ACCIDENT_TRAIN    if is_train else ACCIDENT_VAL
            no_dir   = NO_ACCIDENT_TRAIN if is_train else NO_ACCIDENT_VAL

            for i, item in enumerate(split):
                img   = item.get("image")
                label = item.get("label", item.get("labels", 0))

                if img is None:
                    continue

                # Label 0 = normal, 1 = collision/accident, 2 = deformed
                if label in (1, 2):
                    out_path = acc_dir / f"hf_cad_{split_name}_{i}.jpg"
                    img.save(out_path)
                    saved_acc += 1
                else:
                    out_path = no_dir / f"hf_cad_{split_name}_{i}.jpg"
                    img.save(out_path)
                    saved_no += 1

        print(f"  ✅ Saved {saved_acc} accident images, {saved_no} normal images")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print("     Try: pip install datasets")


def download_hf_accident_classification():
    """
    Download accident classification dataset from HuggingFace.
    Binary: accident / no_accident
    """
    print("\n[HuggingFace] Downloading accident classification dataset...")
    datasets_to_try = [
        "Nahush/accident_non_accident_image_classification",
        "akashdeepsingh5/accident-non-accident-image-classification",
    ]

    for ds_name in datasets_to_try:
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name)
            print(f"  Loaded: {ds_name}")

            saved_acc = 0
            saved_no  = 0

            for split_name, split in ds.items():
                is_train = "train" in split_name
                acc_dir  = ACCIDENT_TRAIN    if is_train else ACCIDENT_VAL
                no_dir   = NO_ACCIDENT_TRAIN if is_train else NO_ACCIDENT_VAL

                for i, item in enumerate(split):
                    img   = item.get("image")
                    label = item.get("label", 0)
                    if img is None:
                        continue
                    if label == 1:
                        img.save(acc_dir / f"hf_acc_{split_name}_{i}.jpg")
                        saved_acc += 1
                    else:
                        img.save(no_dir / f"hf_acc_{split_name}_{i}.jpg")
                        saved_no += 1

            print(f"  ✅ Saved {saved_acc} accident, {saved_no} normal images from {ds_name}")
            return  # success

        except Exception as e:
            print(f"  ⚠️  {ds_name} failed: {e}")
            continue

    print("  ❌ All HuggingFace accident datasets failed. Continuing...")


# ─── Kaggle datasets (requires ~/.kaggle/kaggle.json) ────────────────────────────

def check_kaggle_auth() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n⚠️  Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("   To use Kaggle datasets:")
        print("   1. Go to https://www.kaggle.com/account → 'Create New Token'")
        print("   2. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("   3. Re-run this script")
        return False
    return True


def download_kaggle_dataset(owner: str, dataset: str, dest: Path, unzip_dir: str = None):
    """Download a Kaggle dataset and extract it."""
    try:
        import kaggle
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[Kaggle] Downloading {owner}/{dataset}...")
        kaggle.api.dataset_download_files(f"{owner}/{dataset}", path=str(dest), unzip=True)
        print(f"  ✅ Downloaded to {dest}")
        return True
    except ImportError:
        print("  ❌ kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def process_kaggle_accident_dataset(raw_dir: Path):
    """
    Move images from Kaggle accident detection dataset into our directory structure.
    Expects: raw_dir/Accident/ and raw_dir/Non Accident/ (or similar)
    """
    import shutil as sh
    for subdir in raw_dir.rglob("*"):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if any(k in name for k in ("accident", "crash", "collision")):
            target = ACCIDENT_TRAIN
        elif any(k in name for k in ("no", "normal", "non", "safe")):
            target = NO_ACCIDENT_TRAIN
        else:
            continue

        for img in subdir.glob("*.jpg"):
            sh.copy(img, target / f"kgl_{subdir.parent.name}_{img.name}")
        for img in subdir.glob("*.png"):
            sh.copy(img, target / f"kgl_{subdir.parent.name}_{img.stem}.jpg")


def download_all_kaggle():
    """Download all configured Kaggle datasets."""
    kaggle_datasets = [
        # (owner, dataset, description)
        ("arnav3107",   "accident-detection-dataset", "~3K accident/no-accident frames"),
        ("sartajbhuvaji", "car-accident-dataset",     "~2K labeled accident images"),
        ("khotijaoks",  "car-damages-detection",       "~4K damaged vehicle images"),
    ]

    raw_base = DATA_ROOT / "kaggle_raw"
    for owner, dataset, desc in kaggle_datasets:
        raw_dir = raw_base / dataset
        success = download_kaggle_dataset(owner, dataset, raw_dir)
        if success:
            print(f"  Processing {desc}...")
            process_kaggle_accident_dataset(raw_dir)


# ─── Dataset statistics ──────────────────────────────────────────────────────────

def print_stats():
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    for split in ["train", "val"]:
        acc  = len(list((IMG_DIR / "accident"    / split).glob("*.jpg")))
        no   = len(list((IMG_DIR / "no_accident" / split).glob("*.jpg")))
        total = acc + no
        ratio = f"{acc}/{no}" if total > 0 else "0/0"
        print(f"  {split:6s}: {total:5d} images  (accident/normal: {ratio})")

    print(f"\n  Minimum recommended for fine-tuning: 500 images per class")
    if total > 0:
        acc_total = len(list((IMG_DIR / "accident").rglob("*.jpg")))
        if acc_total < 500:
            print(f"  ⚠️  Only {acc_total} accident images — consider adding more data")
        else:
            print(f"  ✅ Sufficient data for fine-tuning ({acc_total} accident images)")


# ─── Main ────────────────────────────────────────────────────────────────────────

def download_video_urls(url_file: Path, dest_dir: Path):
    """Download video clips from a URL list into dest_dir."""
    try:
        import requests
+        from urllib.parse import urlparse
    except ImportError:
        print("  ❌ requests is required to download videos. Run: pip install requests")
        return

    if not url_file.exists():
        print(f"  ⚠️  URL list not found: {url_file}")
        return

    with url_file.open("r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print(f"  ⚠️  No URLs found in {url_file}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Internet] Downloading videos from {url_file} to {dest_dir}...")

    for i, url in enumerate(urls, start=1):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            parsed = urlparse(url)
            name = os.path.basename(parsed.path) or f"video_{i}.mp4"
            if not name.lower().endswith(".mp4"):
                name = f"{Path(name).stem}_{i}.mp4"
            out_path = dest_dir / name
            with open(out_path, "wb") as out_file:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)
            print(f"  ✅ {out_path.name}")
        except Exception as e:
            print(f"  ⚠️  Failed to download {url}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download AccidentAI training datasets")
    parser.add_argument("--hf-only",  action="store_true", help="Download HuggingFace datasets only (no Kaggle)")
    parser.add_argument("--list",     action="store_true", help="List available datasets and exit")
    parser.add_argument("--videos",   action="store_true", help="Download internet videos listed in training/video_urls.txt")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        print("  HuggingFace (free):")
        print("    - M-ArslanArshad/Car_Accidents_and_deformation_dataset (~4K images)")
        print("    - Nahush/accident_non_accident_image_classification (~2K images)")
        print("  Kaggle (requires API key):")
        print("    - arnav3107/accident-detection-dataset (~3K images)")
        print("    - sartajbhuvaji/car-accident-dataset (~2K images)")
        print("    - khotijaoks/car-damages-detection (~4K images)")
        print("  Video URLs:")
        print("    - training/video_urls_accident.txt")
        print("    - training/video_urls_no_accident.txt")
        return

    print("=" * 50)
    print("  AccidentAI — Dataset Downloader")
    print("=" * 50)

    setup_dirs()

    # Always download HuggingFace (free)
    download_hf_car_accident_deformation()
    download_hf_accident_classification()

    # Internet video downloads
    if args.videos:
        download_video_urls(DATA_ROOT / "video_urls_accident.txt", VIDEO_ACCIDENT)
        download_video_urls(DATA_ROOT / "video_urls_no_accident.txt", VIDEO_NO_ACCIDENT)

    # Kaggle (optional)
    if not args.hf_only:
        if check_kaggle_auth():
            download_all_kaggle()
        else:
            print("\n⏭️  Skipping Kaggle datasets (no API key). Use --hf-only to suppress this message.")

    print_stats()
    print("\n✅ Done! Now run: python training/train_image_classifier.py")


if __name__ == "__main__":
    main()
