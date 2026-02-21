# %%
# After installation, comment this cell and restart the kernel.
# Uses the current Python interpreter (sys.executable) to install into the active environment.

# import sys
# print(sys.executable)
# !{sys.executable} -m pip install pandas pillow imagehash matplotlib scikit-learn torch torchvision imbalanced-learn timm

# %%
# import necessary libraries
import pandas as pd
from pathlib import Path
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms
from torchvision import models
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import timm
import numpy as np
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import RandAugment
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# %% [markdown]
# ### Prepare Metadata CSV

# %%
ROOT = Path("./Classes")
# Some paths have depth of 4 instead of 3, need to list them all out
files = [f for f in ROOT.rglob("*") if f.is_file() and not f.name.startswith(".")]
print("Total files found:", len(files))

# %%
pd.set_option('display.max_colwidth', 100)

# %%
data = []

# %%
# This cell may take a long time to run, to speed up, we can directly read from the processed mushroom_paths.csv file
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

df = pd.DataFrame(
    data,
    columns=["file_path", "depth", "file_name", "file_ext", "width", "height", "num_pixels", "phash"]
)

df["aspect_ratio"] = df["width"] / df["height"]

# %%
# This above cell may take a long time to run, to speed up, we can directly read from the processed mushroom_paths.csv file
if len(data) != 0:
    df.to_csv("mushroom_paths.csv", index=False)
else:
    df = pd.read_csv("mushroom_paths.csv")

# %%
df.shape

# %%
df.head()

# %%
print("File Depth Distribution:")
print(df.groupby('depth').size())

# %%
print("\nSample files with depth 3:")
df[df['depth'] == 3].head()

# %%
print("\nSample files with depth 4:")
df[df['depth'] == 4].head()

# %%
df[df['depth'] == 4]['file_path'].apply(lambda x: x.split('/')[:-1]).value_counts().head()

# %% [markdown]
# `Mushrooms` is redundant in the paths; the directory immediately after `Mushrooms/` represents the species label.

# %%
df['class'] = df['file_path'].apply(lambda x: x.split('/')[1])
df['species'] = df.apply(lambda x: x['file_path'].split('/')[3] if x['depth'] == 4 else x['file_path'].split('/')[2], axis=1)

# %%
df[df['depth'] == 3].head()

# %%
df[df['depth'] == 4].head()

# %%
df['class'].value_counts(normalize=True)

# %% [markdown]
# ### EDA

# %% [markdown]
# #### Basic Quality Checks

# %%
df.head(), df.shape

# %%
# Completeness / quality
print("Missing values per column:")
display(df.isna().sum().sort_values(ascending=False).head(10))

print("\nUnique classes:", df["class"].nunique())
print("Unique species:", df["species"].nunique())
print("File ext counts:\n", df["file_ext"].value_counts().head())

# %% [markdown]
# #### Class Distribution Plot

# %%
counts = df["class"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values)

plt.title("Class Distribution (Counts)")
plt.xlabel("Class")
plt.ylabel("Number of Images")

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.show()

# %%
plt.figure(figsize=(6, 6))

plt.pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Class Distribution (Percentage)")
plt.show()

# %%
class_dist_table = (
    counts
    .rename("count")
    .to_frame()
)

class_dist_table["percentage"] = (
    class_dist_table["count"] / class_dist_table["count"].sum() * 100
).round(2)

class_dist_table


# %% [markdown]
# #### Image Size / Resolution distribution

# %%
plt.figure()
df["width"].hist(bins=50)
plt.title("Image Width Distribution")
plt.xlabel("Width (px)")
plt.ylabel("Count")
plt.show()

plt.figure()
df["height"].hist(bins=50)
plt.title("Image Height Distribution")
plt.xlabel("Height (px)")
plt.ylabel("Count")
plt.show()

plt.figure()
df["num_pixels"].hist(bins=50)
plt.title("Image Resolution (num_pixels) Distribution")
plt.xlabel("Width * Height")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# #### Aspect Ratio + Scatter (spot outliers)

# %%
plt.figure()
df["aspect_ratio"].hist(bins=50)
plt.title("Aspect Ratio Distribution")
plt.xlabel("width/height")
plt.ylabel("Count")
plt.show()

# Spot weird sizes/outliers
plt.figure()
plt.scatter(df["width"], df["height"], s=3)
plt.title("Width vs Height (Outlier Check)")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()


# %% [markdown]
# #### Duplicate Check (Exact and Near)

# %% [markdown]
# Duplicate and near-duplicate images were identified and removed to reduce redundancy and prevent df leakage between training and evaluation sets.

# %% [markdown]
# ##### Exact Duplicate Removal
# - Perceptual hash (pHash) values were used to identify exact duplicate images.
# - Images sharing the same pHash were considered visually identical.
# - For each group of exact duplicates, only the image with the highest resolution (largest number of pixels) was retained.
# - Lower-resolution duplicates were removed from the dataset.

# %%
df_sorted = df.sort_values(["phash", "num_pixels"], ascending=[True, False])

exact_dedup = df_sorted.drop_duplicates(subset=["phash"], keep="first").copy().reset_index(drop=True)

removed = len(df) - len(exact_dedup)
print(f"Rows before: {len(df):,}")
print(f"Rows after : {len(exact_dedup):,}")
print(f"Removed    : {removed:,} ({removed/len(df)*100:.2f}%)")

exact_dedup.to_csv("mushroomDedupExact.csv", index=False)
print("Saved:", "mushroomDedupExact.csv")

# %% [markdown]
# ##### Near-Duplicate Removal
# - Near-duplicate images were identified by computing the Hamming distance between pHash values.
# - Images with a Hamming distance below a predefined threshold were considered near-duplicates.
# - When near-duplicate images were detected, the image with the highest resolution was retained, and remaining lower-resolution images were removed.

# %%
HAMMING_THRESHOLD = 5     # 5 means "very similar"
PREFIX_LEN = 4            # bucket by first 4 hex chars; increase to reduce comparisons

# %%
# Convert to ImageHash
hashes = df["phash"].astype(str).apply(imagehash.hex_to_hash)

# %%
# Bucket indices by hash prefix to avoid O(N^2)
buckets = defaultdict(list)
for i, h in enumerate(hashes):
    buckets[str(h)[:PREFIX_LEN]].append(i)

keep = set(range(len(df)))
dropped = set()

# %%
# Within each bucket, drop near-duplicates by keeping highest resolution
for _, idxs in buckets.items():
    if len(idxs) <= 1:
        continue

    idxs_sorted = sorted(idxs, key=lambda i: df.loc[i, "num_pixels"], reverse=True)

    chosen = []
    for i in idxs_sorted:
        if i in dropped:
            continue
        is_near_dup = any((hashes[i] - hashes[j]) <= HAMMING_THRESHOLD for j in chosen)
        if is_near_dup:
            dropped.add(i)
        else:
            chosen.append(i)

# %%
keep = sorted(list(keep - dropped))
near_dedup = df.iloc[keep].copy().reset_index(drop=True)

print(f"Rows before: {len(df):,}")
print(f"Rows after : {len(near_dedup):,}")
print(f"Removed    : {len(df)-len(near_dedup):,} ({(len(df)-len(near_dedup))/len(df)*100:.2f}%)")

near_dedup.to_csv("mushroomDedupNear.csv", index=False)
print("Saved:", "mushroomDedupNear.csv")

# %%
BASE_DIR = Path(".") 

print("raw :", df.shape)
print("exact:", exact_dedup.shape)
print("near :", near_dedup.shape)

# %% [markdown]
# ##### Stat after Deduplication

# %%
def exact_dup_stats(df):
    vc = df["phash"].astype(str).value_counts()
    dup_hashes = (vc > 1).sum()
    dup_images = vc[vc > 1].sum()
    return dup_hashes, dup_images

rows = []
for name, d in [("raw", df), ("after_exact", exact_dedup), ("after_near", near_dedup)]:
    dup_hashes, dup_images = exact_dup_stats(d)
    rows.append({
        "dataset": name,
        "n_rows": len(d),
        "hashes_with_count>1": dup_hashes,
        "images_in_exact_dup_groups": int(dup_images),
        "exact_dup_rate": float(dup_images / len(d))
    })

summary = pd.DataFrame(rows)
summary


# %% [markdown]
# ##### Sample of Exact Duplicates Removed

# %%
def show_exact_duplicate_group(df, base_dir=BASE_DIR, max_show=6):
    # Count pHash occurrences
    vc = df["phash"].astype(str).value_counts()
    dup_hashes = vc[vc > 1]

    if len(dup_hashes) == 0:
        print("No exact duplicates found.")
        return

    # Take the first duplicate hash as an example
    ph = dup_hashes.index[0]
    group = df[df["phash"].astype(str) == ph].copy()

    # Keep highest-resolution images first
    group = group.sort_values("num_pixels", ascending=False).head(max_show)

    print(f"Example exact-duplicate pHash: {ph}")
    display(group[["file_path", "width", "height", "num_pixels", "class", "species"]])

    plt.figure(figsize=(14, 3))
    for i, (_, row) in enumerate(group.iterrows(), start=1):
        img_path = (base_dir / row["file_path"]).resolve()

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, len(group), i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f'{row["width"]}×{row["height"]}')

    plt.show()


# %%
show_exact_duplicate_group(df)

# %% [markdown]
# ##### Sample of Near Duplicates Removed

# %%

def hamming_hex(h1, h2):
    # robust popcount (works on older Python too)
    x = int(h1, 16) ^ int(h2, 16)
    return bin(x).count("1")

def near_dedup_pairs(df, prefix_len=PREFIX_LEN, thresh=HAMMING_THRESHOLD):
    ph = df["phash"].astype(str).tolist()
    
    buckets = defaultdict(list)
    for idx, h in enumerate(ph):
        buckets[h[:prefix_len]].append(idx)

    # return example near-dup pairs (i, j, distance)
    pairs = []
    for idxs in buckets.values():
        if len(idxs) <= 1:
            continue
        # compare within bucket (still can be big, so we limit example output)
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                d = hamming_hex(ph[i], ph[j])
                if d <= thresh and d != 0:  # d==0 are exact duplicates
                    pairs.append((i, j, d))
                    if len(pairs) >= 30:  # limit to keep it fast
                        return pairs
    return pairs

pairs = near_dedup_pairs(df)
print("Found near-duplicate pairs (sample):", len(pairs))
pairs[:5]

# %%
def show_near_duplicate_example(df, pairs, base_dir=BASE_DIR):
    if not pairs:
        print("No near-duplicate pairs found (or sampling limit reached).")
        return
    
    i, j, d = pairs[0]
    r1 = df.iloc[i]
    r2 = df.iloc[j]
    print("Example near-duplicate pair")
    print("Hamming distance:", d)
    display(pd.DataFrame([r1, r2])[["file_path", "width", "height", "num_pixels", "class", "species", "phash"]])

    plt.figure(figsize=(8, 4))
    for k, row in enumerate([r1, r2], start=1):
        p = base_dir / row["file_path"]
        img = Image.open(p).convert("RGB")
        plt.subplot(1, 2, k)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f'{row["width"]}x{row["height"]}')
    plt.suptitle(f"Near-duplicate example (Hamming={d})")
    plt.show()


# %%
show_near_duplicate_example(df, pairs)

# %% [markdown]
# ##### Summary

# %%
dup_counts = df["phash"].value_counts()
num_dup_hashes = (dup_counts > 1).sum()
num_dup_images = dup_counts[dup_counts > 1].sum()

print("Hashes that appear >1 time:", num_dup_hashes)
print("Images involved in duplicates:", num_dup_images)
print("Duplicate image rate:", num_dup_images / len(df))


# %% [markdown]
# #### Sample Images per Class (visual sanity check)

# %%
def show_samples(df, class_name, n=6):
    sub = df[df["class"] == class_name].sample(n=min(n, len(df[df["class"] == class_name])), random_state=0)
    plt.figure(figsize=(12, 4))
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        plt.subplot(1, n, i)
        img = Image.open(row["file_path"]).convert("RGB")
        plt.imshow(img)
        plt.axis("off")
        plt.title(row["species"][:14])
    plt.suptitle(f"Samples: {class_name}")
    plt.show()

for c in df["class"].unique():
    show_samples(df, c, n=6)

# %%
counts = near_dedup["class"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values)

plt.title("Class Distribution (Counts) After De-Duplication")
plt.xlabel("Class")
plt.ylabel("Number of Images")

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.show()

# %%
near_dedup['class'].value_counts(normalize=True)

# %% [markdown]
# #### Handling Severe Class Imbalance
# 
# The **deadly** class represents less than 1% of the total dataset. Such extreme imbalance introduces several issues, such as:
# 
# * The model may fail to learn meaningful patterns from the **deadly** class as there are not enough images for training.
# * Oversampling the minority class could lead to overfitting.
# * Undersampling the majority classes will result in significantly less data available for training and testing.
# 
# Thus, we are going to combine the **deadly** class and the **poisonous** class into a single **toxic** category to improve class balance and model stability.

# %%
near_dedup['original_class'] = near_dedup['class']
near_dedup['class'] = near_dedup['class'].apply(lambda x: 'toxic' if x == 'poisonous' or x == 'deadly' else x)

# %%
near_dedup['class'].value_counts(normalize=True)

# %%
near_dedup['original_class'].value_counts(normalize=True)

# %%
near_dedup.to_csv("mushroomDedupNearMerged.csv", index=False)

# %% [markdown]
# ### Data Pipeline & Transformations

# %% [markdown]
# #### Standardization
# 
# Standardization was applied dynamically during df loading to ensure consistent input format for model training. All images were resized to a fixed input resolution to accommodate convolutional neural network requirements. Pixel values were scaled to the range [0, 1] and normalized using predefined channel-wise mean and standard deviation values. This process was deterministic and applied consistently across training, validation, and test datasets.

# %% [markdown]
# #### Augmentation
# Data augmentation was applied only to the training dataset to improve model robustness and generalization. Random transformations, including horizontal flipping, rotation, and color jittering, were used to introduce controlled variability in image appearance while preserving semantic content. Augmentation was performed at load time and did not modify the original image files.

# %% [markdown]
# #### Balanced Dataset

# %% [markdown]
# ##### [TODO]

# %%
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(near_dedup[['file_path']], near_dedup[['class']])

df_balanced = X_res.copy()
df_balanced["class"] = y_res

# %%
df_balanced["class"].value_counts()

# %%
IMG_SIZE = 224

# RGB statistics of ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# MILDER IMG AUGMENTATION FOR RESNET-18.
train_tfms_mild = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),  # converts to [0,1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# STRONGER IMG AUGMENTATION FOR RESNET-50 and SWIN. UNCOMMENT AND RUN IF TESTING RESNET-50 and SWIN
train_tfms_strong = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),  # stronger jitter
        RandAugment(num_ops=2, magnitude=9),  # learned augmentation policy
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),  # CutOut-style regularization
    ]
)

# Run for either model
val_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# %% [markdown]
# ### Models

# %% [markdown]
# #### Preparing Train, Val, Test Datasets & Loaders

# %%
train, temp = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['class'])
test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['class'])

CLASSES = sorted(train["class"].unique().tolist())

train.shape, val.shape, test.shape

# %%
train['class'].value_counts(normalize=True)

# %%
test['class'].value_counts(normalize=True)

# %%
val['class'].value_counts(normalize=True)

# %%
# Set device for PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# %%
from MushroomDataset import *

# %%
train_class_to_idx = {c: i for i, c in enumerate(CLASSES)}
train_class_to_idx

# %%
# For ResNet-18, use mild augmentation. 
train_ds_mild = MushroomDataset(train, transform=train_tfms_mild, class_to_idx=train_class_to_idx, classes=CLASSES)
# For ResNet-50 and SWIN, use stronger augmentation. 
train_ds_strong = MushroomDataset(train, transform=train_tfms_strong, class_to_idx=train_class_to_idx, classes=CLASSES)
val_ds = MushroomDataset(val, transform=val_tfms, class_to_idx=train_class_to_idx, classes=CLASSES) # ensure same mapping
test_ds = MushroomDataset(test, transform=val_tfms, class_to_idx=train_class_to_idx, classes=CLASSES) # ensure same mapping

# %%
BATCH_SIZE = 32
NUM_WORKERS = 4

# %%
# For ResNet-18
train_loader_mild = DataLoader(train_ds_mild, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
# For ResNet-50 and SWIN
train_loader_strong = DataLoader(train_ds_strong, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

# %% [markdown]
# #### Baseline Model - ResNet18

# %% [markdown]
# ##### Training and Validation

# %%
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# %%
def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, is_train=False, return_preds=False):
    running_loss = 0.0
    running_correct = 0
    total = 0

    d = device if isinstance(device, torch.device) else torch.device(device)
    use_amp = (d.type == "cuda") and (scaler is not None)

    all_preds, all_labels = [], []

    if is_train:
        model.train()
        for img, labels in loader:
            img = img.to(d, non_blocking=True)
            labels = labels.to(d, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(True):
                    outputs = model(img)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(img)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(1)
            running_loss += loss.item() * img.size(0)
            running_correct += (preds == labels).sum().item()
            total += img.size(0)

            if return_preds:
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
    else:
        model.eval()
        with torch.no_grad():
            for img, labels in loader:
                img = img.to(d, non_blocking=True)
                labels = labels.to(d, non_blocking=True)

                outputs = model(img)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(1)
                running_loss += loss.item() * img.size(0)
                running_correct += (preds == labels).sum().item()
                total += img.size(0)

                if return_preds:
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

    total_loss = running_loss / max(total, 1)
    total_acc  = running_correct / max(total, 1)

    if return_preds:
        return total_loss, total_acc, total, torch.cat(all_labels), torch.cat(all_preds)
    else:
        return total_loss, total_acc, total, None, None

# %%
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=3):
    best_val_loss = float('inf')
    bad_epochs = 0

    d = device if isinstance(device, torch.device) else torch.device(device)
    use_amp = (d.type == "cuda")
    print("device = ", d)
    print("use_amp = ", use_amp)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    
    for epoch in range(num_epochs):
        print(f"\n>>> Epoch {epoch + 1} start", flush=True)

        train_loss, train_acc, _, _, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, d, scaler=scaler, is_train=True, return_preds=False
        )
        val_loss, val_acc, _, _, _ = run_one_epoch(
            model, val_loader, criterion, optimizer, d, is_train=False, return_preds=False
        )

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> Saved best model", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.", flush=True)
                break

    return model


# %%
train(model, train_loader_mild, val_loader, criterion, optimizer, num_epochs=5, device=device, patience=5)

# %%
# rebuild model architecture and load best model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# %% [markdown]
# ##### Testing

# %%
test_loss, test_acc, test_total, test_labels, test_preds = run_one_epoch(
    model, test_loader, criterion, optimizer, device, is_train=False, return_preds=True
)

# %%
print("Test accuracy:", test_acc)
print("Balanced accuracy:", balanced_accuracy_score(test_labels, test_preds))

# %%
cm = confusion_matrix(test_labels, test_preds)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n",
    classification_report(test_labels, test_preds, target_names=CLASSES, digits=4))

# %% [markdown]
# #### 2nd Model - ResNet50

# %% [markdown]
# #### 3rd Model - Swin Transformer Tiny

# %%
### first run

# %%
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=len(CLASSES)
)
model = model.to(device)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# %%
num_epochs = 20
best_acc = 0.0

for epoch in range(num_epochs):
    # TRAIN
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total
    
    # VALIDATE
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'swin_tiny_best.pth')
        print(f"✓ Saved best model (Acc: {best_acc:.4f})")
    
    scheduler.step()

# %%
model.load_state_dict(torch.load('swin_tiny_best.pth'))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = test_correct / test_total
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

# %%
# second run w/ updates

# %%
# ============================================================
# SWIN TINY -- MODEL
# ============================================================
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=len(CLASSES))
model = model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

# %%
# ============================================================
# SWIN TINY -- DATASETS & LOADERS (strong augmentation)
# ============================================================
train_ds = MushroomDataset(train, transform=train_tfms_strong, class_to_idx=train_class_to_idx, classes=CLASSES)
val_ds   = MushroomDataset(val,   transform=val_tfms,          class_to_idx=train_class_to_idx, classes=CLASSES)
test_ds  = MushroomDataset(test,  transform=val_tfms,          class_to_idx=train_class_to_idx, classes=CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

# %%
# ============================================================
# SWIN TINY -- PHASE 1: TRAIN HEAD ONLY
# ============================================================
print("\n--- Swin Tiny Phase 1: Training head only ---")
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)
HEAD_EPOCHS = 3

for epoch in range(HEAD_EPOCHS):
    model.train()
    running_loss, correct, total_n = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Swin Phase1 {epoch+1}/{HEAD_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_n += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"  Phase1 Epoch {epoch+1} - Loss: {running_loss/len(train_loader.dataset):.4f}, Acc: {correct/total_n:.4f}")


# %%
# ============================================================
# SWIN TINY -- PHASE 2: FULL FINE-TUNING
# ============================================================
print("\n--- Swin Tiny Phase 2: Full fine-tuning ---")
for param in model.parameters():
    param.requires_grad = True

param_groups = [
    {"params": model.head.parameters(), "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": 1e-5},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

num_epochs = 20
warmup_epochs = 2
scheduler = SequentialLR(optimizer,
    schedulers=[LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)],
    milestones=[warmup_epochs])

best_acc = 0.0
patience = 5
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Swin Epoch {epoch+1}/{num_epochs} Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Swin Epoch {epoch+1}/{num_epochs} Val"):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total

    print(f"\nSwin Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'swin_tiny_best.pth')
        print(f"  Saved best model (Val Acc: {best_acc:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    scheduler.step()

# %%
# ============================================================
# SWIN TINY -- TEST EVALUATION
# ============================================================
model.load_state_dict(torch.load('swin_tiny_best.pth', map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Swin Testing"):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\n{'='*60}")
print(f"TEST RESULTS: Swin Tiny")
print(f"{'='*60}")
print(f"Overall Accuracy:  {(all_preds == all_labels).mean():.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix - Swin Tiny')
plt.tight_layout()
plt.savefig('swin_tiny_confusion_matrix.png', dpi=150)
plt.show()

print("\nSafety-Critical Recall:")
for cls in ['poisonous', 'deadly']:
    if cls in CLASSES:
        i = CLASSES.index(cls)
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {cls:20s} recall = {rec:.4f}  ({cm[i,i]}/{cm[i].sum()})")

# %%
%pip install grad-cam

# %%
# ============================================================
# SWIN TINY -- GRAD-CAM INTERPRETABILITY
# ============================================================
# pip install pytorch-grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load best model
model.load_state_dict(torch.load('swin_tiny_best.pth', map_location=device))
model.eval()

# For Swin Tiny, target the last normalization layer before the head
target_layers = [model.layers[-1].blocks[-1].norm2]

cam = GradCAM(model=model, target_layers=target_layers)

# Inverse normalization to get displayable images
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1.0/s for s in IMAGENET_STD]
)

# --- Visualize correct predictions (sanity check: model looking at mushroom?) ---
print("=== Grad-CAM on CORRECT predictions ===")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

shown = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct_mask = predicted.eq(labels)

        for i in range(images.size(0)):
            if not correct_mask[i]:
                continue
            if shown >= 16:
                break

            input_tensor = images[i].unsqueeze(0)
            

            target_class = labels[i].item()

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(target_class)])[0]

            rgb_img = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            row, col = divmod(shown, 4)
            axes[row, col].imshow(cam_image)
            axes[row, col].set_title(f"True: {CLASSES[target_class]}\nPred: {CLASSES[predicted[i].item()]}",
                                      fontsize=9)
            axes[row, col].axis('off')
            shown += 1

        if shown >= 16:
            break

plt.suptitle("Grad-CAM - Correct Predictions (should focus on mushroom body)", fontsize=14)
plt.tight_layout()
plt.savefig('swin_tiny_gradcam_correct.png', dpi=150)
plt.show()

# %%
# --- Visualize MISCLASSIFIED predictions (where is the model confused?) ---
print("\n=== Grad-CAM on MISCLASSIFIED predictions ===")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

shown = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        wrong_mask = ~predicted.eq(labels)

        for i in range(images.size(0)):
            if not wrong_mask[i]:
                continue
            if shown >= 16:
                break

            input_tensor = images[i].unsqueeze(0)
            true_class = labels[i].item()
            pred_class = predicted[i].item()

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(pred_class)])[0]

            rgb_img = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            row, col = divmod(shown, 4)
            axes[row, col].imshow(cam_image)
            axes[row, col].set_title(f"True: {CLASSES[true_class]}\nPred: {CLASSES[pred_class]}",
                                      fontsize=9, color='red')
            axes[row, col].axis('off')
            shown += 1

        if shown >= 16:
            break

if shown < 16:
    for idx in range(shown, 16):
        row, col = divmod(idx, 4)
        axes[row, col].axis('off')

plt.suptitle("Grad-CAM - Misclassified (check if model focuses on background/wrong cues)", fontsize=14)
plt.tight_layout()
plt.savefig('swin_tiny_gradcam_misclassified.png', dpi=150)
plt.show()

# %%
# --- Safety-critical: Grad-CAM on poisonous/deadly samples ---
print("\n=== Grad-CAM on poisonous/deadly samples ===")
safety_classes = [c for c in ['poisonous', 'deadly'] if c in CLASSES]
fig, axes = plt.subplots(len(safety_classes), 5, figsize=(20, 4 * len(safety_classes)))
if len(safety_classes) == 1:
    axes = [axes]

for row_idx, cls_name in enumerate(safety_classes):
    cls_idx = CLASSES.index(cls_name)
    shown = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            mask = labels.eq(cls_idx)

            for i in range(images.size(0)):
                if not mask[i]:
                    continue
                if shown >= 5:
                    break

                input_tensor = images[i].unsqueeze(0)
                outputs = model(input_tensor)
                _, pred = outputs.max(1)
                pred_class = pred.item()

                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=[ClassifierOutputTarget(pred_class)])[0]

                rgb_img = inv_normalize(images[i].cpu()).permute(1, 2, 0).numpy()
                rgb_img = np.clip(rgb_img, 0, 1)

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                color = 'green' if pred_class == cls_idx else 'red'
                axes[row_idx][shown].imshow(cam_image)
                axes[row_idx][shown].set_title(
                    f"True: {cls_name}\nPred: {CLASSES[pred_class]}",
                    fontsize=9, color=color)
                axes[row_idx][shown].axis('off')
                shown += 1

            if shown >= 5:
                break

    for j in range(shown, 5):
        axes[row_idx][j].axis('off')

plt.suptitle("Grad-CAM - Safety-Critical Classes (poisonous/deadly)", fontsize=14)
plt.tight_layout()
plt.savefig('swin_tiny_gradcam_safety.png', dpi=150)
plt.show()

# %%



