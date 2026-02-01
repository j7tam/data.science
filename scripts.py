import pandas as pd
from pathlib import Path
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

ROOT = Path("Classes")
# Some paths have depth of 4 instead of 3, need to list them all out
files = [f for f in ROOT.rglob("*") if f.is_file() and not f.name.startswith(".")]
print("Total files found:", len(files))

data = []
for f in files:
    try:
        with Image.open(f) as img:
            w, h = img.size
            img_hash = imagehash.phash(img)
    except Exception:
        continue

    data.append([
        str(f),
        len(f.relative_to(ROOT).parts),
        f.stem,
        f.suffix.lstrip("."),
        w,
        h,
        w * h,
        img_hash
    ])

data = pd.DataFrame(
    data,
    columns=["file_path", "depth", "file_name", "file_ext", "width", "height", "num_pixels", "phash"]
)

data["aspect_ratio"] = data["width"] / data["height"]

data['class'] = data['file_path'].apply(lambda x: x.split('/')[1])
data['species'] = data.apply(lambda x: x['file_path'].split('/')[3] if x['depth'] == 4 else x['file_path'].split('/')[2], axis=1)

data.to_csv("mushroom_paths.csv", index=False)