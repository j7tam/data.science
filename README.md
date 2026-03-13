
# Mushroom Edibility Classification with Deep Learning

**DSC 288R Capstone Project**  
University of California, San Diego

A computer vision project that trains deep learning models to classify mushrooms into **edible**, **conditionally edible**, and **toxic** categories using image data.

---

# Step 1: Clone the Repository

Clone the repository **in Terminal**:

```bash
git clone https://github.com/j7tam/data.science.git
cd data.science
```

All commands below should be run **in Terminal**, from the **repository root directory**:

```text
data.science/
```

---

# Step 2: Download the Dataset

The dataset is too large to store in this repository, so it must be downloaded separately.

Dataset source: 
https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset

---

## Step 2.1: Install Kaggle CLI

Install Kaggle CLI **in Terminal**:

```bash
pip install kaggle
```

Verify installation **in Terminal**:

```bash
kaggle --version
```

---

## Step 2.2: Configure Kaggle API Credentials

Go to this page **in your browser**: https://www.kaggle.com/settings

Under **API**, click **Create New Token**.

Then run the following commands **in Terminal**:

1. Create the Kaggle configuration directory:

    ```bash
    mkdir -p ~/.kaggle
    ```

2. Copy your Kaggle username and API token, then create the credentials file (replace `YOUR_KAGGLE_USERNAME` and `YOUR_KAGGLE_API_TOKEN` with your actual values):

    ```bash
    echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_TOKEN"}' > ~/.kaggle/kaggle.json
    ```

3. Set permissions:

    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

4. Verify the setup **in Terminal**:

    ```bash
    kaggle datasets list
    ```

---

## Step 2.3: Download the Dataset

From the **repository root directory**, run **in Terminal**:

```bash
chmod +x download_data.sh
./download_data.sh
```

The script will:

1. Download the dataset from Kaggle
2. Extract the files
3. Move the images into:

```text
Classes/
```

---

## Expected Folder After Download

After running the dataset download script, the repository should contain:

```text
Classes/
├── edible/
├── conditionally_edible/
├── poisonous/
└── deadly/
```

The CSV files in `Data/splits/` reference these image paths.

---

## Dataset Overview

The original mushroom dataset contains four classes:

* edible
* conditionally edible
* poisonous
* deadly

To reduce class imbalance, the two most dangerous classes are merged:

```text
poisonous + deadly → toxic
```

Final modeling classes:

* edible
* conditionally edible
* toxic

Dataset splits used for training and evaluation are stored in `Data/splits/`.

---

# Step 3:Install Required Tools

Before running the setup scripts, make sure the following tools are installed.

## 1. Conda (required)

This project requires **Conda** because the environment setup script uses `conda` commands.

You can install Conda using either **Anaconda** or **Miniconda**.

### Install Anaconda

https://www.anaconda.com/download

### Install Miniconda

https://docs.conda.io/en/latest/miniconda.html

After installation, verify Conda **in Terminal**:

```bash
conda --version
```

## 2. Git

Git is needed to clone the repository.

### Install Git

https://git-scm.com/install/

After installation, verify the installation **in Terminal**:

```bash
git --version
```

## 3. Jupyter

You do **not** need to install Jupyter manually.

`setup_env.sh` will automatically install:

* `jupyterlab`
* `ipykernel`

and register the notebook kernel for this project.

---

# Step 4: Environment Setup

From the **repository root directory**, run **in Terminal**:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

The script will:

1. Remove any existing `mushroom-cls` environment
2. Create the environment using `environment.yml`
3. Install JupyterLab and ipykernel
4. Register a Jupyter kernel named:

```text
Python (mushroom-cls)
```
You can add a short note that users may launch notebooks from an IDE as well. Here is a clean, README-ready version:

---

# Step 5: Launch Jupyter

After the environment is created, run the following **in Terminal**, from the **repository root directory**:

```bash
conda run -n mushroom-cls python -m jupyter lab
```

Then, **in Jupyter Notebook / JupyterLab**:

1. Open the notebook you want to run
2. Select the kernel:

```text
Python (mushroom-cls)
```

3. Run the notebook cells

---

## Step 5 Alternative: Use an IDE

You may also open the notebooks using an IDE such as **VS Code**, **PyCharm**, or the classic **Jupyter Notebook** interface.

If using an IDE:

1. Open the repository folder
2. Open the notebook file (`.ipynb`)
3. Select the kernel:

```text
Python (mushroom-cls)
```

4. Run the cells normally

> ⚠️ Make sure the IDE is using the `mushroom-cls` Conda environment as the interpreter.

---


# Workflow

## 01_Preprocessing

Used for:

* data cleaning
* duplicate removal
* train / validation / test split generation

## 02_EDA

Used for:

* dataset statistics
* class balance visualization
* image distribution analysis

## 03_DataPipelineAndTransformers

Used for:

* image transforms
* dataset class
* DataLoader setup

## 04_Model

Contains model training and evaluation notebooks:

* **ResNet18 baseline**
* **ResNet18 toxic-priority training**
* **ResNet50 experiments**
* **Swin Tiny Transformer**

Pretrained weights are stored in:

```text
04_Model/best_models/
```

> **Note**: Only some pretrained model weights are stored in the repository due to GitHub file size limits. If pretrained weights are missing, you can train the models using the notebooks in `04_Model/` or contact the project team for access to the full set of pretrained weights.

---

# Running Training / Evaluation

## In Terminal

Start JupyterLab from the **repository root directory**:

```bash
conda run -n mushroom-cls python -m jupyter lab
```

## In Notebook

A typical workflow is:

1. Load dataset splits from `Data/splits/`
2. Create transforms from `Utils/transforms.py`
3. Build dataloaders using `Utils/MushroomDataset.py`
4. Train or evaluate models using notebooks in `04_Model/`

Example for loading a pretrained model **in a notebook cell**:

```python
import torch

model.load_state_dict(
    torch.load("04_Model/best_models/resnet18_best.pth")
)

model.eval()
```

---

# Reproducibility

This project supports reproducibility by:

* providing fixed dataset splits
* using `environment.yml` for dependency management
* providing pretrained model weights
* using consistent data pipelines

---

# Contributors

Project Group 8

* Chiu-Chiu (JoJo) Lin
* Nicholas Shor
* Joanna Tam
* [Xuewen Yang](https://www.linkedin.com/in/xuewen-daphne-yang/)

University of California, San Diego
DSC 288R Capstone Project

---

# License

This project is intended for **academic use only**.
