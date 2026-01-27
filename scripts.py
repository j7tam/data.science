import pandas as pd
from pathlib import Path

ROOT = Path("Classes")

files = list(ROOT.glob("*/*/*"))

mushroom_paths = pd.DataFrame({
    "path": [str(f) for f in files],
    "species": [f.parent.name for f in files],
    "class": [f.parent.parent.name for f in files],
})
mushroom_paths.to_csv("mushroom_paths.csv")

