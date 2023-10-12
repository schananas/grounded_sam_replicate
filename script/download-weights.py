#!/usr/bin/env python

import os
import subprocess
import sys

sys.path.insert(0, "/weights/Grounded-Segment-Anything")
sys.path.insert(0, "/weights/Grounded-Segment-Anything/GroundingDINO")

os.mkdir("/weights")
# Setting up the current directory
os.chdir("/weights")

# Downloading the file using wget module from shell
subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)

# Cloning the Github repo
subprocess.run(['git', 'clone', 'https://github.com/IDEA-Research/Grounded-Segment-Anything'], check=True)

# Changing the current path
os.chdir("/weights/Grounded-Segment-Anything")

# Installing the required modules
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)

# Changing to other paths and installing modules
folders = ['GroundingDINO', 'segment_anything']
for folder in folders:
  os.chdir(f"/weights/Grounded-Segment-Anything/{folder}")
  subprocess.run([sys.executable, '-m', 'pip', 'install', '.'], check=True)

# Coming back to the main working directory
os.chdir("/weights/Grounded-Segment-Anything")

import torch
from huggingface_hub import hf_hub_download
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

sam_checkpoint = '/weights/sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
