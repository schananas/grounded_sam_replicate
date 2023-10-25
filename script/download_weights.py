#!/usr/bin/env python

import os
import subprocess
import shutil

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)
# Setting up the current directory
os.chdir("weights")

# Downloading the file using wget module from shell
subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)

# Cloning the Github repo
subprocess.run(['git', 'clone', 'https://github.com/IDEA-Research/GroundingDINO'], check=True)
subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/segment-anything'], check=True)
