from torch.utils.data import Dataset
from PIL import Image
import torch

class MushroomDataset(Dataset):
    def __init__(self, df, transform=None, label_col="class", class_to_idx=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["file_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            
        y = self.class_to_idx[row[self.label_col]]
        return img, torch.tensor(y, dtype=torch.long)
