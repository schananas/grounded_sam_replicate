#!/usr/bin/env python

import os
import subprocess
import shutil
from huggingface_hub import hf_hub_download

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


# Download huggingface models
def download_model_hf(repo_id, filename, ckpt_config_filename):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, cache_dir="/src/weights/")
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="/src/weights/")
    return cache_config_file, cache_file

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

# Download the models
cache_config_file, cache_file = download_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

with open('hf_path_exports.py', 'w') as f:
    f.write(f"cache_config_file = r'{cache_config_file}'\n")
    f.write(f"cache_file = r'{cache_file}'\n")

