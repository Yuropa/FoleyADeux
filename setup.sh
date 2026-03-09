#!/bin/bash

# 1. Create a fresh environment with Python and FFmpeg
echo "Creating Conda environment 'foley'..."
conda create -y -n foley python=3.10 pip setuptools wheel

# 2. Activate the environment (the hook makes it work inside a script)
eval "$(conda shell.bash hook)"
conda activate foley
conda install -c conda-forge "ffmpeg=*=*gpl*" -y

# 3. Install the correct PyTorch
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
fi

# 4. Install standard Pip packages safely
echo "Installing pip requirements..."
pip install -r requirements.txt

# 5. Fetch and install local submodules
echo "Updating git submodules..."
git submodule update --init --recursive

echo "Installing local editable packages..."
pip install -e ./external/video-foley/RMS_ControlNet_Inference
pip install -e ./external/video-foley/RMS_ControlNet_Inference/AudioLDMControlNetInfer/Model/AudioLdm
pip install -e ./external/video-foley/RMS_ControlNet_Inference/TorchJAEKWON

echo "Setup Complete! Run 'conda activate foley' to get started."

echo "Downloading model checkpoints..."
mkdir -p ./ckpt
HF_BASE="https://huggingface.co/jnwnlee/video-foley/resolve/main"
for FILE in checkpoint_000500_Video2RMS.pt ControlNetstep300000.pth opts.yml; do
    if [ ! -f "./ckpt/$FILE" ]; then
        echo "  Downloading $FILE..."
        curl -L --retry 3 "$HF_BASE/$FILE" -o "./ckpt/$FILE"
    else
        echo "  $FILE already exists, skipping."
    fi
done

echo "Preparing audio dataset..."

DATA_DIR="/Data"
TARGET_DIR_PARENT="/Data"
TARGET_DIR="$TARGET_DIR_PARENT/GreatestHits"
ZIP_FILE="$DATA_DIR/vis-data.zip"
URL="https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip"

# If dataset already exists, skip download & extraction
if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    echo "Dataset already exists at $TARGET_DIR, skipping download and extraction."
else
    # Make sure data directory exists
    mkdir -p "$TARGET_DIR"
    mkdir -p "$DATA_DIR"

    echo "Downloading audio dataset (~50GB)..."
    curl -L -C - "$URL" -o "$ZIP_FILE" || { echo "Download failed, skipping."; }

    echo "Extracting dataset..."
    unzip -q "$ZIP_FILE" -d "$TARGET_DIR" || { echo "Extraction failed, skipping."; }

    mv "$TARGET_DIR_PARENT/vis-data/"* "$TARGET_DIR"/
    rm -rf "$TARGET_DIR_PARENT/vis-data"

    echo "Cleaning up zip file..."
    rm -f "$ZIP_FILE"

    echo "Dataset ready at $TARGET_DIR"
fi
