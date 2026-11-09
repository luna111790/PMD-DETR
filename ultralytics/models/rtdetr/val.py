# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = 'RTDETRValidator',  


class RTDETRDataset(YOLODataset):
    






    def __init__(self, *args, data=None, **kwargs):
        
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    
    def load_image(self, i, rect_mode=False):
        
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            
            transforms = Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class RTDETRValidator(DetectionValidator):
    



















    def build_dataset(self, img_path, mode='val', batch=None):
        







        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  
            hyp=self.args,
            rect=False,  
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds):
        
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  
            
            
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  
            
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  

        return outputs

    def update_metrics(self, preds, batch):
        
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[..., [0, 2]] *= shape[1] / self.args.imgsz  
            predn[..., [1, 3]] *= shape[0] / self.args.imgsz  

            
            if nl:
                tbox = ops.xywh2xyxy(bbox)  
                tbox[..., [0, 2]] *= shape[1]  
                tbox[..., [1, 3]] *= shape[0]  
                labelsn = torch.cat((cls, tbox), 1)  
                
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  

            
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)
