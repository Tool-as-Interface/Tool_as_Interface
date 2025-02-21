import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
import torch.nn.functional as F
import supervision as sv
import matplotlib.pyplot as plt
from PIL import Image
from groundingdino.util.inference import Model
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
from ti.utils.Grounded_SAM.object_info import ObjectInfo 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "ti/utils/Grounded_SAM/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CONFIG_PATH_B = "ti/utils/Grounded_SAM/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "ti/utils/Grounded_SAM/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CHECKPOINT_PATH_B = "ti/utils/Grounded_SAM/groundingdino_swinb_cogcoor.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "ti/utils/Grounded_SAM/sam_vit_h_4b8939.pth"
SAM_HQ_CHECKPOINT_PATH = "ti/utils/Grounded_SAM/sam_hq_vit_h.pth"



def initialize_segmentation_models():
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH_B, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH_B)


    use_sam_hq = True
    # Building SAM Model and SAM Predictor
    if use_sam_hq:
        sam_predictor = SamPredictor(sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_HQ_CHECKPOINT_PATH).to(DEVICE))
    else:
        sam_predictor = SamPredictor(sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE))

    return grounding_dino_model, sam_predictor

def release_segmentation_models(grounding_dino_model, sam_predictor):
    # Explicitly delete models and release memory
    del grounding_dino_model
    del sam_predictor
    torch.cuda.empty_cache()  # Clear unused GPU memory


def compute_iou(box1, box2):
    """
    Compute the intersection-over-union (IoU) of two bounding boxes.
    box1, box2: Bounding boxes in the format [x1, y1, x2, y2].
    Returns the IoU value.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def get_segmentation_mask(image, segmentation_labels: list, saved_path: str, grounding_dino_model, sam_predictor, debug=False, save_img=True, remove_segmented=False):
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.8
    IOU_THRESHOLD = 0.5  # Threshold for removing overlapping boxes
    
    detected_image = image.copy()
    
    # Initialize empty image to store the final results
    final_image = np.zeros_like(image)
    

    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    # Accumulate detections and labels for final annotation
    all_labels = []


    # Detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=detected_image,
        classes=segmentation_labels,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    if len(detections.xyxy) == 0:
        return None, image  

    # NMS post-process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Filter out overlapping boxes
    non_overlapping_indices = []
    for i in range(len(detections.xyxy)):
        is_overlapping = False
        for j in range(len(non_overlapping_indices)):
            if compute_iou(detections.xyxy[i], detections.xyxy[non_overlapping_indices[j]]) > IOU_THRESHOLD:
                is_overlapping = True
                break
        if not is_overlapping:
            non_overlapping_indices.append(i)

    # Keep only non-overlapping detections
    detections.xyxy = detections.xyxy[non_overlapping_indices]
    detections.confidence = detections.confidence[non_overlapping_indices]
    detections.class_id = detections.class_id[non_overlapping_indices]

    if debug:

        labels = [
            f"{segmentation_labels[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                        detections=detections,
                                        labels=labels)

        # save the annotated grounding dino image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

    sam_predictor.set_image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    result_masks = []
    for box in detections.xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
        
    detections.mask = np.array(result_masks)
    # detections.class_id[:] = class_id
    
    # Combine all masks for the current label
    combined_mask = np.zeros(detections.mask[0].shape, dtype=bool)
    for mask in detections.mask:
        combined_mask = np.maximum(combined_mask, mask)
                
    if remove_segmented:
        final_image = image.copy()
        final_image[combined_mask] = 0  # Set the segmented areas to black (or transparent if needed)
    else:
        all_labels = [
            f"{segmentation_labels[int(class_id)]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]

        # merged_detections = sv.Detections.merge(detections)
        final_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        final_image = label_annotator.annotate(scene=final_image,
                                    detections=detections,
                                    labels=all_labels)

    if save_img:
        cv2.imwrite(saved_path, final_image)  # Save the annotated image (optional)
    return combined_mask, final_image


if __name__ == "__main__":
    image_path = "./test_image.png"
    saved_path = "./save_image.png"
    image = cv2.imread(image_path)

    
    grounding_dino_model, sam_predictor = initialize_segmentation_models()
    
    segmentation_labels = ['human arms.sleeve.shirt.pants.hoody.jacket.robot.kinova.ur5e']
    masks, _ = get_segmentation_mask(image, segmentation_labels, saved_path, grounding_dino_model, sam_predictor, remove_segmented=True)
