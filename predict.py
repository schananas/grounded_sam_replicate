import os
import shutil
from typing import Iterator

import torch
from cog import BasePredictor, Input, Path
from compel import Compel
from diffusers.utils import load_image
import numpy as np



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        print("setup...")



    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Image",
                default=None,
            ),
            prompt: str = Input(
                description="Positive mask prompt",
                default="clothes,shoes",
            ),
            negative_prompt: str = Input(
                description="Negative mask prompt",
                default="pants",
            ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        print("run prediction......")
        yield Path(image)


            # output_path = f"/tmp/seed-{this_seed}.png"
            # output.images[0].save(output_path)
            # yield Path(output_path)
            # result_count += 1


