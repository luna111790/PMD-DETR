import argparse
from runtime import ROOT, load_model, project_path

def main():
    parser = argparse.ArgumentParser(description='Visualize PMD-DETR detections.')
    parser.add_argument('--source', required=True)
    parser.add_argument('--weights', default='weights/best.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--line-width', type=int, default=3)
    parser.add_argument('--labels', action='store_true')
    cli = parser.parse_args()
    model = load_model(cli.weights)
    for _ in model.predict(source=str(project_path(cli.source)), imgsz=640,
                           device=cli.device, conf=cli.conf, save=True,
                           save_txt=True, save_conf=True, show_labels=cli.labels,
                           show_conf=cli.labels, line_width=cli.line_width,
                           project=str(ROOT / 'runs' / 'predict'), name='detections',
                           exist_ok=False, stream=True):
        pass

if __name__ == '__main__':
    main()
