# Ultralytics YOLO 🚀, AGPL-3.0 license


from pathlib import Path

import cv2
import numpy as np
import torch


_imshow = cv2.imshow  


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    









    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    










    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    






    _imshow(winname.encode('unicode_escape').decode(), mat)



_torch_save = torch.save  


def torch_save(*args, **kwargs):
    






    try:
        import dill as pickle  
    except ImportError:
        import pickle

    if 'pickle_module' not in kwargs:
        kwargs['pickle_module'] = pickle  
    return _torch_save(*args, **kwargs)
