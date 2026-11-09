# Ultralytics YOLO 🚀, AGPL-3.0 license

























import glob
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch.cuda

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ASSETS, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from ultralytics.utils.checks import check_requirements, check_yolo
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device


def benchmark(model=WEIGHTS_DIR / 'yolov8n.pt',
              data=None,
              imgsz=160,
              half=False,
              int8=False,
              device='cpu',
              verbose=False):
    

























    import pandas as pd
    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():  
        emoji, filename = '❌', None  
        try:
            assert i != 9 or LINUX, 'Edge TPU export only supported on Linux'
            if i == 10:
                assert MACOS or LINUX, 'TF.js export only supported on macOS and Linux'
            elif i == 11:
                assert sys.version_info < (3, 11), 'PaddlePaddle export only supported on Python<=3.10'
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'

            
            if format == '-':
                filename = model.ckpt_path or model.cfg
                exported_model = model  
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, int8=int8, device=device, verbose=False)
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), 'export failed'
            emoji = '❎'  

            
            assert model.task != 'pose' or i != 7, 'GraphDef Pose inference is not supported'
            assert i not in (9, 10), 'inference not supported'  
            assert i != 5 or platform.system() == 'Darwin', 'inference only supported on macOS>=10.13'  
            exported_model.predict(ASSETS / 'bus.jpg', imgsz=imgsz, device=device, half=half)

            
            data = data or TASK2DATA[model.task]  
            key = TASK2METRIC[model.task]  
            results = exported_model.val(data=data,
                                         batch=1,
                                         imgsz=imgsz,
                                         plots=False,
                                         device=device,
                                         half=half,
                                         int8=int8,
                                         verbose=False)
            metric, speed = results.results_dict[key], results.speed['inference']
            y.append([name, '✅', round(file_size(filename), 1), round(metric, 4), round(speed, 2)])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f'Benchmark failure for {name}: {e}'
            LOGGER.warning(f'ERROR ❌️ Benchmark failure for {name}: {e}')
            y.append([name, emoji, round(file_size(filename), 1), None, None])  

    
    check_yolo(device=device)  
    df = pd.DataFrame(y, columns=['Format', 'Status❔', 'Size (MB)', key, 'Inference time (ms/im)'])

    name = Path(model.ckpt_path).name
    s = f'\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n'
    LOGGER.info(s)
    with open('benchmarks.log', 'a', errors='ignore', encoding='utf-8') as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  
        floor = verbose  
        assert all(x > floor for x in metrics if pd.notna(x)), f'Benchmark failure: metric(s) < floor {floor}'

    return df


class ProfileModels:
    























    def __init__(self,
                 paths: list,
                 num_timed_runs=100,
                 num_warmup_runs=10,
                 min_time=60,
                 imgsz=640,
                 half=True,
                 trt=True,
                 device=None):
        












        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  
        self.device = device or torch.device(0 if torch.cuda.is_available() else 'cpu')

    def profile(self):
        
        files = self.get_files()

        if not files:
            print('No matching *.pt or *.onnx files found.')
            return

        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix('.engine')
            if file.suffix in ('.pt', '.yaml', '.yml'):
                model = YOLO(str(file))
                model.fuse()  
                model_info = model.info()
                if self.trt and self.device.type != 'cpu' and not engine_file.is_file():
                    engine_file = model.export(format='engine',
                                               half=self.half,
                                               imgsz=self.imgsz,
                                               device=self.device,
                                               verbose=False)
                onnx_file = model.export(format='onnx',
                                         half=self.half,
                                         imgsz=self.imgsz,
                                         simplify=True,
                                         device=self.device,
                                         verbose=False)
            elif file.suffix == '.onnx':
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output

    def get_files(self):
        
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ['*.pt', '*.onnx', '*.yaml']
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {'.pt', '.yaml', '.yml'}:  
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        print(f'Profiling: {sorted(files)}')
        return [Path(file) for file in sorted(files)]

    def get_onnx_model_info(self, onnx_file: str):
        


        
        return 0.0, 0.0, 0.0, 0.0

    def iterative_sigma_clipping(self, data, sigma=2, max_iters=3):
        
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        
        model = YOLO(engine_file)
        input_data = np.random.rand(self.imgsz, self.imgsz, 3).astype(np.float32)  

        
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time

        
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed['inference'])  

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  
        return np.mean(run_times), np.std(run_times)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        


        check_requirements('onnxruntime')
        import onnxruntime as ort

        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  
        sess = ort.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])

        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type

        
        if 'float16' in input_type:
            input_dtype = np.float16
        elif 'float' in input_type:
            input_dtype = np.float32
        elif 'double' in input_type:
            input_dtype = np.float64
        elif 'int64' in input_type:
            input_dtype = np.int64
        elif 'int32' in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f'Unsupported ONNX datatype {input_type}')

        input_data = np.random.rand(*input_tensor.shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], {input_name: input_data})
            elapsed = time.time() - start_time

        
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        
        layers, params, gradients, flops = model_info
        return f'| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ± {t_engine[1]:.2f} ms | {params / 1e6:.1f} | {flops:.1f} |'

    def generate_results_dict(self, model_name, t_onnx, t_engine, model_info):
        
        layers, params, gradients, flops = model_info
        return {
            'model/name': model_name,
            'model/parameters': params,
            'model/GFLOPs': round(flops, 3),
            'model/speed_ONNX(ms)': round(t_onnx[0], 3),
            'model/speed_TensorRT(ms)': round(t_engine[0], 3)}

    def print_table(self, table_rows):
        
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
        header = f'| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |'
        separator = '|-------------|---------------------|--------------------|------------------------------|-----------------------------------|------------------|-----------------|'

        print(f'\n\n{header}')
        print(separator)
        for row in table_rows:
            print(row)
