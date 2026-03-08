#!/bin/bash

echo "Downloading Mushroom Classification Dataset from Kaggle..."

# check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI not found."
    echo "Install it with: pip install kaggle"
    exit
fi

# create dataset folder
mkdir -p Classes

# download dataset
kaggle datasets download -d zedsden/mushroom-classification-dataset

# unzip dataset
unzip mushroom-classification-dataset.zip -d Classes

# remove zip file
rm mushroom-classification-dataset.zip

echo "Dataset downloaded and extracted to ./Classes"