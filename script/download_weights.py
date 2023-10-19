#!/usr/bin/env python

import os
import sys
import shutil

sys.path.insert(0, "/src/weights/Grounded-Segment-Anything")
sys.path.insert(0, "/src/weights/Grounded-Segment-Anything/GroundingDINO")

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)
# Setting up the current directory
os.chdir("weights")
#
# # Downloading the file using wget module from shell
# subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
#
# # Cloning the Github repo
# subprocess.run(['git', 'clone', 'https://github.com/IDEA-Research/Grounded-Segment-Anything'], check=True)




