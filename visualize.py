import argparse
from pathlib import Path
from runtime import ROOT, initialize, project_path, bind_source_classes

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM++ heatmaps.')
    parser.add_argument('--source', required=True)
    parser.add_argument('--weights', default='weights/best.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--output', default='runs/gradcam')
    parser.add_argument('--boxes', action='store_true')
    cli = parser.parse_args()
    initialize()
    import numpy as np
    import torch
    import visualization.gradcam as cam
    torch.manual_seed(0)
    np.random.seed(0)
    original_load = cam.attempt_load_weights
    def load(*args, **kwargs):
        return bind_source_classes(original_load(*args, **kwargs))
    cam.attempt_load_weights = load
    heatmap = cam.rtdetr_heatmap(weight=str(project_path(cli.weights)),
        device=cli.device, method='GradCAMPlusPlus', layer=[16,20,23,26],
        backward_type='all', conf_threshold=cli.conf, ratio=1.0,
        show_box=cli.boxes, renormalize=False)
    source = project_path(cli.source)
    output = project_path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.iterdir()) if source.is_dir() else [source]
    files = [p for p in files if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']]
    if not files:
        raise ValueError('No images found')
    try:
        for path in files:
            target = output / (path.stem + '.png')
            if target.exists():
                raise FileExistsError(target)
            heatmap.process(str(path), str(target))
            if not target.is_file():
                raise RuntimeError('Heatmap generation failed for ' + path.name)
            print(target)
    finally:
        heatmap.method.activations_and_grads.release()

if __name__ == '__main__':
    main()
