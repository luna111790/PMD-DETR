import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs' / '.settings'))

def initialize():
    import builtins
    import dill
    import torch
    builtins.NoneType = type(None)
    dill._dill._reverse_typemap['NoneType'] = type(None)
    torch.set_num_threads(4)

def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def dataset_config(value='dataset/data.yaml'):
    import yaml
    source = project_path(value)
    data = yaml.safe_load(source.read_text(encoding='utf8'))
    base = Path(data.get('path') or '.').expanduser()
    base = (source.parent / base).resolve() if not base.is_absolute() else base.resolve()
    data['path'] = str(base)
    for split in ('train', 'val', 'test'):
        entries = data.get(split)
        if entries is None:
            continue
        paths = entries if isinstance(entries, list) else [entries]
        resolved = []
        for entry in paths:
            path = Path(entry).expanduser()
            path = path.resolve() if path.is_absolute() else (base / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f'{split} dataset path does not exist: {path}')
            resolved.append(str(path))
        data[split] = resolved if isinstance(entries, list) else resolved[0]
    target = ROOT / 'runs' / 'dataset.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf8')
    return str(target)

def bind_source_classes(model):
    import importlib
    for module in model.modules():
        cls = type(module)
        if cls.__module__ in ['ultralytics.nn.extra_modules.PFG', 'ultralytics.nn.extra_modules.DMSSP']:
            source = importlib.import_module(cls.__module__)
            replacement = getattr(source, cls.__name__, None)
            if replacement is None and cls.__name__ == 'Branch':
                replacement = source.PFGA.Branch
            if replacement is None:
                raise TypeError(cls.__qualname__)
            module.__class__ = replacement
    return model

def load_model(weights='weights/best.pt'):
    initialize()
    from ultralytics import RTDETR
    model = RTDETR(str(project_path(weights)))
    bind_source_classes(model.model)
    return model
