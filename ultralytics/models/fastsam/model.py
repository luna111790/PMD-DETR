# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    











    def __init__(self, model='FastSAM-x.pt'):
        
        if str(model) == 'FastSAM.pt':
            model = 'FastSAM-x.pt'
        assert Path(model).suffix not in ('.yaml', '.yml'), 'FastSAM models only support pre-trained models.'
        super().__init__(model=model, task='segment')

    @property
    def task_map(self):
        
        return {'segment': {'predictor': FastSAMPredictor, 'validator': FastSAMValidator}}
