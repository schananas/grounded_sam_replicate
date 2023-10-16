import os
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO


def detect(image,image_src, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_src, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return annotated_frame, boxes

def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
        )
    return masks.cpu()

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def run_grounding_sam(local_image_path, positive_promt, negative_promt):
    image_source, image = load_image(local_image_path)

    annotated_frame, detected_boxes = detect(image, image_source, positive_promt, groundingdino_model)
    neg_annotated_frame, neg_detected_boxes = detect(image, image_source, negative_promt, groundingdino_model)

    combined_frame = np.hstack((annotated_frame, neg_annotated_frame))

    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)

    neg_segmented_frame_masks = segment(image_source, sam_predictor, boxes=neg_detected_boxes)


    # Merging all positive masks
    merged_mask = np.logical_or.reduce(segmented_frame_masks[:,0])

    # Merging all negative masks
    merged_neg_mask = np.logical_or.reduce(neg_segmented_frame_masks[:,0])

    # Annotation using merged positive mask
    annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)

    # Annotation using merged positive mask
    final_annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)

    # Annotation using merged negative mask
    neg_annotated_frame_with_mask = draw_mask(merged_neg_mask, neg_annotated_frame)

    # Converting positive mask into PIL image
    mask = (merged_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition
    neg_mask = (merged_neg_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition

    # Use logical operations to subtract the negative mask from the original mask
    final_subtracted_mask = mask & ~neg_mask

    # Update inverted mask definition
    final_subtracted_inverted_mask = 255 - final_subtracted_mask


    return Image.fromarray(final_annotated_frame_with_mask), Image.fromarray(neg_annotated_frame_with_mask), Image.fromarray(final_subtracted_mask), Image.fromarray(final_subtracted_inverted_mask)
