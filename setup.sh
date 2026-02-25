#!/bin/bash

# Make sure the submodule is there
git submodule init
git submodule update

conda create -n foley python=3.11 -y
conda activate foley
conda install pip -y
conda install ffmpeg=6.1.0 x264 -y
pip install -r requirements.txt

conda install git-lfs -y
git clone https://huggingface.co/jnwnlee/video-foley ./ckpt
