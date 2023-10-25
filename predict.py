import os
import sys
import subprocess
import json
from typing import Any
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel
import uuid

from subprocess import call

os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'
os.environ['AM_I_DOCKER'] = 'true'
os.environ['BUILD_WITH_CUDA'] = 'true'

env_vars = os.environ.copy()


HOME = os.getcwd()

sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO")
sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything")

os.chdir("/src/weights/Grounded-Segment-Anything/GroundingDINO")
call("echo $CUDA_HOME", shell=True, env=env_vars)
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir(HOME)

os.chdir("/src/weights/Grounded-Segment-Anything/segment_anything")
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir(HOME)

from typing import Iterator
from huggingface_hub import hf_hub_download
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")

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
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

        sam_checkpoint = '/src/weights/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Image",
                default="https://st.mngbcn.com/rcs/pics/static/T5/fotos/outfit/S20/57034757_56-99999999_01.jpg",
            ),
            mask_prompt: str = Input(
                description="Positive mask prompt",
                default="clothes,shoes",
            ),
            negative_mask_prompt: str = Input(
                description="Negative mask prompt",
                default="pants",
            ),
            adjustment_factor: int = Input(
                description="Mask Adjustment Factor (-ve for erosion, +ve for dilation)",
                default=0,
            ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())

        print(f"Running prediction: {predict_id}...")

        annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = run_grounding_sam(image,
                                                                                                    mask_prompt,
                                                                                                    negative_mask_prompt,
                                                                                                    self.groundingdino_model,
                                                                                                    self.sam_predictor,
                                                                                                    adjustment_factor)
        variable_dict = {
            'annotated_picture_mask': annotated_picture_mask,
            'neg_annotated_picture_mask': neg_annotated_picture_mask,
            'mask': mask,
            'inverted_mask': inverted_mask
        }

        output_dir = "/tmp/" + predict_id
        os.makedirs(output_dir, exist_ok=True)  # create directory if it doesn't exist

        for var_name, img in variable_dict.items():
            random_filename = output_dir + "/" + var_name + ".jpg"
            rgb_img = img.convert('RGB')  # Converting image to RGB
            rgb_img.save(random_filename)
            yield Path(random_filename)