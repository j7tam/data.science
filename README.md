# Mushroom Edibility Classification with Deep Learning

**DSC 288R Capstone Project**  
University of California, San Diego

A computer vision project that trains deep learning models to classify mushrooms into **edible**, **conditionally edible**, and **toxic** categories using image data.

---

# Repository Structure

```
data.science/
├── Data/                          # CSV splits + metadata
│   ├── mushroomBalanced.csv
│   ├── mushroomDedupExact.csv
│   ├── mushroomDedupNear.csv
│   ├── mushroomDedupNearMerged.csv
│   ├── mushroomDetailedPaths.csv
│   ├── mushroomPaths.csv
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── 01_Preprocessing/               # Data cleaning and split generation
├── 02_EDA/                         # Exploratory data analysis
├── 03_DataPipelineAndTransformers/ # Data pipeline + transforms
├── 04_Model/                       # Training + evaluation notebooks
│   ├── best_models/
│   │   ├── resnet18_best.pth
│   │   └── resnet18_toxic_priority_best.pth
│   ├── 04_01_SplitDataset.ipynb
│   ├── 04_02_ResNet18.ipynb
│   ├── 04_02_ResNet18ToxicPriority.ipynb
│   ├── 04_03_RESNET50.ipynb
│   └── 04_04_SwinTinyTransformer.ipynb
│
├── Graphs/                         # Visualizations and figures
├── Reports/                        # Project reports
├── Utils/                          # Dataset + training utilities
│   ├── constants.py
│   ├── dataLoaders.py
│   ├── MushroomDataset.py
│   ├── project_utils.py
│   └── transforms.py
│
├── download_data.sh                # Script to download dataset
├── requirements.txt
└── README.md
```

---

# Dataset Overview

The original mushroom dataset contains four classes:

- edible
- conditionally edible
- poisonous
- deadly

To reduce class imbalance, the two most dangerous classes are merged:

```
poisonous + deadly → toxic
```

Final modeling classes:

- edible
- conditionally edible
- toxic

Dataset splits used for training and evaluation are stored in:

```
Data/splits/
```

---

# Environment Setup

Create a Python environment and install dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Download the Dataset

The image dataset is **too large to store in this repository**, so it must be downloaded separately.

Dataset source:  
https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset

After cloning the repository, follow the steps below to download the dataset.

---

# Step 1: Install Kaggle CLI

The dataset will be downloaded using the Kaggle API.

Install the Kaggle CLI:

```bash
pip install kaggle
```

Verify installation:

```bash
kaggle --version
```

---

# Step 2: Configure Kaggle API Credentials

1. Go to https://www.kaggle.com/settings

2. Click **Create New API Token**

3. Download the file `kaggle.json`

4. Move the file to the Kaggle config directory:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
```

5. Set the required permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

# Step 3: Download the Dataset

From the **root directory of the repository**, run:

```bash
bash download_data.sh
```

The script will:

1. Download the dataset from Kaggle
2. Extract the dataset
3. Place the images into:

```
Classes/
```

---

# Expected Folder After Download

After running the script, the repository should contain:

```
Classes/
├── edible/
├── conditionally_edible/
├── poisonous/
└── deadly/
```

The CSV files in `Data/splits/` reference these image paths.

---

# Workflow

## 01_Preprocessing

- Data cleaning
- Duplicate removal
- Train/validation/test split generation

## 02_EDA

- Dataset statistics
- Class balance visualization
- Image distribution analysis

## 03_DataPipelineAndTransformers

- Image transforms
- Dataset class
- DataLoader setup

## 04_Model

Model training and evaluation notebooks:

- **ResNet18 baseline**
- **ResNet18 toxic-priority training**
- **ResNet50 experiments**
- **Swin Tiny Transformer**

Pretrained weights are stored in:

```
04_Model/best_models/
```

---

# Running Training / Evaluation

Typical workflow:

1. Load dataset splits from `Data/splits/`
2. Create transforms from `Utils/transforms.py`
3. Build dataloaders using `Utils/MushroomDataset.py`
4. Train models using notebooks in `04_Model/`

Example loading a pretrained model:

```python
import torch

model.load_state_dict(
    torch.load("04_Model/best_models/resnet18_best.pth")
)

model.eval()
```

---

# Reproducibility

This project ensures reproducibility by:

- Providing **fixed dataset splits**
- Including **all dependencies in `requirements.txt`**
- Providing **pretrained model weights**
- Using **consistent data pipelines**

---

# Contributors

Project Group 8

- Chiu-Chiu (JoJo) Lin
- Nicholas Shor
- Joanna Tam
- Xuewen Yang

University of California, San Diego  
DSC 288R Capstone Project

---

# License

This project is intended for academic use.