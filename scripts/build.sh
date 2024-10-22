#!/bin/bash

# create clean environment and setup necessary software
conda create -n cyber python=3.10 && conda activate cyber
conda install pytorch==2.3.0 torchvision==0.18.0  cudatoolkit=11.1 -c pytorch -c nvidia

# Install the cyber package
pip install -e .