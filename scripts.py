import pandas as pd
from pathlib import Path

ROOT = Path("Classes")

files = list(ROOT.glob("*/*/*"))

mushroom_paths = pd.DataFrame({
    "path": [str(f) for f in files],
    "species": [f.parent.name for f in files],
    "class": [f.parent.parent.name for f in files],
    "file_name": [f.name.split(".")[0] for f in files],
    "file_ext": [f.name.split(".")[-1] for f in files]
})
mushroom_paths.to_csv("mushroom_paths.csv", index=False)