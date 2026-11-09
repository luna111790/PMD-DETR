import argparse
import json
from runtime import ROOT, dataset_config, project_path


def main():
    parser = argparse.ArgumentParser(description='Resolve bundled dataset paths to absolute paths.')
    parser.add_argument('--data', default='dataset/data.yaml')
    args = parser.parse_args()
    import yaml
    target = project_path(dataset_config(args.data))
    data = yaml.safe_load(target.read_text(encoding='utf8'))
    print(json.dumps(dict(config=str(target), dataset={key: data.get(key) for key in
          ('path', 'train', 'val', 'test')}, model=str(project_path('configs/pmd_detr.yaml')),
          weights=str(project_path('weights/best.pt')), output=str(ROOT / 'runs')),
          ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
