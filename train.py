import argparse
import json
from runtime import ROOT, initialize, project_path, dataset_config

def main():
    parser = argparse.ArgumentParser(description='Train PMD-DETR from random initialization.')
    parser.add_argument('--config', default='configs/train.json')
    parser.add_argument('--data', default='dataset/data.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--name', default='pmd_detr')
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    initialize()
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    class PMDTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            assert weights is None
            model = super().get_model(cfg, weights, verbose)
            model.nc = self.data['nc']
            model.criterion = model.init_criterion()
            model.criterion.nwd_loss = True
            model.criterion.iou_ratio = 0.5
            model.criterion.nwd_constant = 6.4
            assert sum(p.numel() for p in model.parameters()) == 15245652
            return model

    args = json.loads(project_path(cli.config).read_text(encoding='utf8'))
    model_path = project_path(args.pop('model'))
    args.pop('mode', None)
    args.update(data=dataset_config(cli.data), device=cli.device, workers=cli.workers,
                project=str(ROOT / 'runs' / 'train'), name=cli.name, exist_ok=False)
    for key in ['epochs', 'batch', 'seed']:
        if getattr(cli, key) is not None:
            args[key] = getattr(cli, key)
    if cli.smoke:
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path(args['data']).read_text(encoding='utf8'))
        training_images = Path(data['path']) / data['train']
        if not training_images.is_dir():
            raise ValueError('--smoke requires a training image directory')
        files = sorted(p for p in training_images.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])[:58]
        manifest = ROOT / 'runs' / 'smoke_train.txt'
        manifest.write_text('\n'.join(str(p) for p in files) + '\n', encoding='utf8')
        data['train'] = str(manifest)
        smoke_data = ROOT / 'runs' / 'smoke_data.yaml'
        smoke_data.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf8')
        args.update(epochs=1, fraction=0.02, batch=2, workers=0, cache=False,
                    data=str(smoke_data), project=str(ROOT / 'runs' / 'smoke'), name='train', save_period=-1)
    model = RTDETR(str(model_path))
    model.train(trainer=PMDTrainer, **args)

if __name__ == '__main__':
    main()
