#!/bin/bash
set -e

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

# 4. Install standard pip packages
echo "Installing pip requirements..."
pip install -r requirements.txt

# 5. Install local packages from libs/
echo "Installing local packages..."
pip install -e ./libs/AudioLDM/AudioLDM/Model/AudioLdm
pip install -e ./libs/AudioLDM
pip install -e ./libs/TorchJaekwon

git clone https://github.com/CompVis/taming-transformers.git ./libs/taming-transformers
pip install -e ./libs/taming-transformers

# 6. Download model checkpoints (large binary files, not tracked by git)
echo "Downloading model checkpoints..."
mkdir -p ./ckpt
HF_BASE="https://huggingface.co/jnwnlee/video-foley/resolve/main"
for FILE in checkpoint_000500_Video2RMS.pt ControlNetstep300000.pth; do
    if [ ! -f "./ckpt/$FILE" ]; then
        echo "  Downloading $FILE..."
        curl -L --retry 3 "$HF_BASE/$FILE" -o "./ckpt/$FILE"
    else
        echo "  $FILE already exists, skipping."
    fi
done

echo ""
echo "Setup complete! To start the app:"
echo "  conda activate foley"
echo "  python -m app"
echo "  python -m app --share  # to share on the local network"
