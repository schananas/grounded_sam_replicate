import os
import sys
import subprocess
from subprocess import call
import shutil
from typing import Iterator
import torch
from cog import BasePredictor, Input, Path
from compel import Compel
from diffusers.utils import load_image
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam

sys.path.insert(0, "script")
sys.path.insert(0, "weights/Grounded-Segment-Anything")
sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO")
sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO/groundingdino")
sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything")
sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything/segment_anything")

os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")

        # Changing the current path
        os.chdir("/src/weights/Grounded-Segment-Anything")

        #Installing the required modules
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)

        # Changing to other paths and installing modules
        folders = ['GroundingDINO', 'segment_anything']
        for folder in folders:
          os.chdir(f"/src/weights/Grounded-Segment-Anything/{folder}")
          subprocess.run([sys.executable, '-m', 'pip', 'install', '.'], check=True)

        # Coming back to the main working directory
        # os.chdir("/src/weights/Grounded-Segment-Anything")

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
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        print("run prediction......")
        print("start...")


        call("find / -type d -name cuda 2>/dev/null", shell=True)


        annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = run_grounding_sam(image,
                                                                                                    mask_prompt,
                                                                                                    negative_mask_prompt,
                                                                                                    self.groundingdino_model,
                                                                                                    self.sam_predictor)
        yield Path(annotated_picture_mask)
        yield Path(neg_annotated_picture_mask)
        yield Path(mask)
        yield Path(inverted_mask)

            # output_path = f"/tmp/seed-{this_seed}.png"
            # output.images[0].save(output_path)
            # yield Path(output_path)
            # result_count += 1


