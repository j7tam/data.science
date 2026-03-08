#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mushroom-cls"
ENV_FILE="environment.yml"
KERNEL_NAME="mushroom-cls"
KERNEL_DISPLAY_NAME="Python (mushroom-cls)"

echo "==> Starting environment setup..."

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or not on PATH."
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found in current directory."
  exit 1
fi

echo "==> Removing any existing env named $ENV_NAME (if it exists)..."
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

ENV_PREFIX="$(conda info --base)/envs/$ENV_NAME"
if [ -d "$ENV_PREFIX" ]; then
  echo "==> Removing leftover env directory: $ENV_PREFIX"
  rm -rf "$ENV_PREFIX"
fi

echo "==> Creating conda env from $ENV_FILE ..."
conda env create -f "$ENV_FILE"

echo "==> Verifying Python inside the env ..."
conda run -n "$ENV_NAME" python -V
conda run -n "$ENV_NAME" python -c "import sys; print(sys.executable)"

echo "==> Installing Jupyter and ipykernel ..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install ipykernel jupyterlab

echo "==> Removing old Jupyter kernel (if it exists) ..."
jupyter kernelspec uninstall "$KERNEL_NAME" -f >/dev/null 2>&1 || true

echo "==> Registering Jupyter kernel ..."
conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "==> Available kernels:"
jupyter kernelspec list || true

echo
echo "==> Setup complete."
echo "To start Jupyter, run:"
echo "conda run -n $ENV_NAME python -m jupyter lab"
echo
echo "To activate the env manually, run:"
echo "conda activate $ENV_NAME"