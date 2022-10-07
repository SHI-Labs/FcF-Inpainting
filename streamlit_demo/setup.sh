#!/bin/sh
eval "$(conda shell.bash hook)"
conda env remove --name fcf
conda create --name fcf -y python=3.7
conda activate fcf
conda env list
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip3 install -r requirements.txt