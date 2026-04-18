import os
import shutil
from pathlib import Path

ARCHIVE_PATH = r"C:\Users\anujm\Downloads\archive"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

archive = Path(ARCHIVE_PATH)
part1 = archive / "HAM10000_images_part_1"
part2 = archive / "HAM10000_images_part_2"

print("Copying images from part 1...")
count = 0
for img in part1.glob("*.jpg"):
    shutil.copy(img, IMAGES_DIR / img.name)
    count += 1
print(f"Copied {count} images")

print("Copying images from part 2...")
count = 0
for img in part2.glob("*.jpg"):
    shutil.copy(img, IMAGES_DIR / img.name)
    count += 1
print(f"Copied {count} images")

shutil.copy(archive / "HAM10000_metadata.csv", DATA_DIR / "HAM10000_metadata.csv")
print("Done!")