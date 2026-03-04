from torch.utils.data import Dataset
from PIL import Image
import torch
from pathlib import Path

class MushroomDataset(Dataset):
    def __init__(self, df, transform=None, label_col="class", class_to_idx=None, classes=None, root_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.classes = classes
        self.class_to_idx = class_to_idx or {}

        self.root_dir = (
            Path(root_dir).expanduser().resolve() if root_dir is not None else None
         )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fp = row["file_path"]
        fp = Path(fp)

        # If it's not absolute and root_dir is provided, join them
        if not fp.is_absolute() and self.root_dir is not None:
            fp = self.root_dir / fp

        # Optional: normalize (won't break if file doesn't exist)
        fp = fp.expanduser()

        if not fp.exists():
            raise FileNotFoundError(
                f"Image not found: {fp} (original file_path='{row['file_path']}')"
            )

        img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = self.class_to_idx[row[self.label_col]]
        return img, torch.tensor(y, dtype=torch.long)
