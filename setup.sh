#!/bin/bash

conda create -n foley python=3.9.18 -y  
conda activate foley

conda install pip -y
pip install --upgrade pip wheel

#conda install -c pytorch -y \
#    pytorch==2.1.1 \
 #   torchvision==0.16.1 \
#    torchaudio==2.1.1
#pip install torcheval

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -y
pip install torcheval

conda install ffmpeg=6.1.0 x264 -c conda-forge -y
conda install -c conda-forge -y \
    pillow=10.0.1 \
    pyyaml=6.0.1 \
    numpy \
    scipy \
    scikit-learn \
    opencv \
    ffmpeg=6.1.0 \
    x264 \
    lightning \
    git-lfs \
    moviepy=1.0.3

pip install librosa==0.10.1
pip install yacs==0.1.8
pip install einops
pip install torchvision
pip install h5py
pip install torchlibrosa
pip install transformers
pip install ftfy
pip install braceexpand
pip install pandas
pip install webdataset
pip install wget
pip install torchaudio
pip install timm
pip install matplotlib
pip install av
python -m pip install git+https://github.com/CompVis/taming-transformers

git submodule init
git submodule update

git submodule update --init --recursive

export KMP_DUPLICATE_LIB_OK=TRUE

#conda install lightning -c conda-forge
#pip install -e ./external/video-foley/RMS_ControlNet_Inference
#pip install -e ./external/video-foley/RMS_ControlNet_Inference/AudioLDMControlNetInfer/Model/AudioLdm
#pip install -e ./external/video-foley/RMS_ControlNet_Inference/TorchJAEKWON

echo "Preparing audio dataset..."

DATA_DIR="./data"
TARGET_DIR_PARENT="./mnt"
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

echo "Downloading models..."
git clone https://huggingface.co/jnwnlee/video-foley ./ckpt
