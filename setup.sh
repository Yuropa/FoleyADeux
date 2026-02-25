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

git submodule init
git submodule update

git submodule update --init --recursive

export KMP_DUPLICATE_LIB_OK=TRUE

#conda install lightning -c conda-forge
#pip install -e ./external/video-foley/RMS_ControlNet_Inference
#pip install -e ./external/video-foley/RMS_ControlNet_Inference/AudioLDMControlNetInfer/Model/AudioLdm
#pip install -e ./external/video-foley/RMS_ControlNet_Inference/TorchJAEKWON

git clone https://huggingface.co/jnwnlee/video-foley ./ckpt
