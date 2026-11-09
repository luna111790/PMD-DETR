# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    











    
    h, w = image_shape

    
    boxes[boxes[:, 0] < threshold, 0] = 0  
    boxes[boxes[:, 1] < threshold, 1] = 0  
    boxes[boxes[:, 2] > w - threshold, 2] = w  
    boxes[boxes[:, 3] > h - threshold, 3] = h  
    return boxes


def bbox_iou(box1, boxes, iou_thres=0.9, image_shape=(640, 640), raw_output=False):
    












    boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    
    union = box1_area + box2_area - intersection

    
    iou = intersection / union  
    if raw_output:
        return 0 if iou.numel() == 0 else iou

    
    return torch.nonzero(iou > iou_thres).flatten()
