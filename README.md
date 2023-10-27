[![Replicate](https://replicate.com/schananas/grounded_sam/badge)](https://replicate.com/schananas/grounded_sam/badge)

# Grounded Sam

Implementation of Grounding DINO & Segment Anything, and it allows masking based on prompt, which is useful for programmed inpainting.

This project combines strengths of two different models in order to build a very powerful pipeline for solving complex masking problems.

Segment-Anything aims to segment everything in an image, which needs prompts (as boxes/points/text) to generate masks.

Grounding DINO, a strong zero-shot detector which, is capable of to generate high quality boxes and labels with free-form text.

On top of Segment-Anything & Grounding DINO this project adds possibility to prompt multiple masks and combine them into one, as well to subtract negative mask for fine grain control.

## Citation

```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```