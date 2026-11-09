import argparse
import json
from pathlib import Path
from runtime import ROOT, load_model, dataset_config

def main():
    parser = argparse.ArgumentParser(description='Evaluate PMD-DETR on val or test.')
    parser.add_argument('--weights', default='weights/best.pt')
    parser.add_argument('--data', default='dataset/data.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plots', action='store_true')
    cli = parser.parse_args()
    model = load_model(cli.weights)
    result = model.val(data=dataset_config(cli.data), split=cli.split, imgsz=640,
                       batch=cli.batch, device=cli.device, workers=cli.workers,
                       half=False, plots=cli.plots, save_json=False,
                       project=str(ROOT / 'runs' / 'evaluate'), name=cli.split,
                       exist_ok=False)
    report = dict(split=cli.split, metrics=result.results_dict,
                  names=result.names, per_class_ap50=result.box.ap50.tolist(),
                  per_class_ap=result.box.ap.tolist())
    target = Path(result.save_dir) / 'metrics.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
