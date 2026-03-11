#!/bin/bash
set -euo pipefail

echo "Downloading Mushroom Classification Dataset from Kaggle..."

# check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found."
    echo "Install it with: pip install kaggle"
    exit 1
fi

ZIP_NAME="mushroom-classification-dataset.zip"
TMP_DIR="tmp_mushroom_download"

# clean temp dir
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

# download dataset into temp dir
kaggle datasets download -d zedsden/mushroom-classification-dataset -p "$TMP_DIR"

# unzip into temp dir
unzip -q "$TMP_DIR/$ZIP_NAME" -d "$TMP_DIR"

# remove old Classes if needed
rm -rf Classes/

# move the actual Classes folder to project root
mv "./$TMP_DIR/mushroom_dataset/Classes" .

# clean up
rm -rf "$TMP_DIR"

echo "Dataset downloaded and extracted to ./Classes"