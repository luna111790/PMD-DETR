# PMD-DETR

本工程包含 C=6.4 主模型的训练代码、训练完成的 best 权重、完整数据集，以及评估、检测可视化和 Grad-CAM++ 入口。

## 环境安装

使用 Python 3.9；GPU 运行环境使用支持 CUDA 的 PyTorch 2.0.1。

```bash
conda create -n pmd-detr python=3.9 -y
conda activate pmd-detr
pip install -r requirements.txt
```

从本工程根目录运行下列命令。直接使用工程内的 `ultralytics` 源码，无需另外安装 Ultralytics。

## 文件结构

```text
configs/pmd_detr.yaml   主模型结构
configs/train.json     训练参数
weights/best.pt        已完成训练的 C=6.4 主模型
dataset/data.yaml      数据集配置
dataset/train/         2,900 张训练图像及对应标注
dataset/val/           307 张原始验证图像及对应标注
dataset/test/          743 张原始测试图像及对应标注
ultralytics/           原模型与训练框架源码
visualization/         原 Grad-CAM 实现
train.py               训练入口
evaluate.py            验证、测试入口
predict.py             检测框可视化入口
visualize.py           Grad-CAM++ 入口
runtime.py             路径、权重加载与兼容设置
prepare_paths.py       生成并显示数据集的绝对路径
```

数据和权重均为实际文件，不依赖外部目录或软链接。运行结果统一写入 `runs/`。

## 路径设置

工程可以解压到任意目录，不需要填写原机器的目录。数据集始终包含在本工程的 `dataset/` 中。

```bash
python prepare_paths.py
```

此命令将包内数据集位置解析为当前机器上的绝对路径，写入 `runs/dataset.yaml`，其中 `path`、`train`、`val`、`test` 四项均为绝对路径，并检查对应目录是否存在。训练和评估入口会自动执行相同的解析，无需手工运行此命令。

随包提供的 `dataset/data.yaml` 是可搬移的路径配置：`path: .` 相对于该配置文件所在目录，三个分区分别为 `train/images`、`val/images` 和 `test/images`。程序实际使用的是生成后的绝对路径配置。工程移动或在另一台机器上解压后，重新运行入口即可生成新位置的绝对路径，不沿用旧路径。

模型配置、权重、检测图像、热图输入和结果目录也统一根据工程根目录解析为绝对路径。命令行中的包内相对路径始终相对于工程根目录，不依赖终端当前目录；也可直接传入绝对路径。

`runs/`、缓存、临时日志和机器相关设置不随工程包分发。

## 训练

```bash
python train.py --device 0
```

默认从随机初始化开始训练：seed=0、输入 640×640、batch=16、AdamW、FP32、最多 200 epochs、patience=50。NWD 常数为 6.4，回归损失权重为 L1:GIoU:NWD=5:1:1。

原有网络结构、前向计算、损失实现和训练框架保持不变。训练参数来自主模型的实际训练配置。`weights/best.pt` 用于评估和推理，不作为训练初始化。

```bash
python train.py --device 0 --batch 16 --workers 8 --seed 0 --name pmd_seed0
```

训练结果位于 `runs/train/<运行名>/`，权重位于其 `weights/best.pt`。同名输出会自动使用新的编号目录。

快速检查训练环境：

```bash
python train.py --smoke --device 0
```

此命令使用 58 张训练图像运行 1 个 epoch，完成反向传播、优化器更新、保存权重及验证。结果写入 `runs/smoke/`，不会覆盖随工程提供的主模型权重。

## 验证与测试

验证随附 best 权重：

```bash
python evaluate.py --split val --device 0
```

主模型在 307 张原始验证图像上的参考结果为 AP50=90.09%、AP50–95=71.90%。

运行独立测试集并输出性能曲线和混淆矩阵：

```bash
python evaluate.py --split test --device 0 --plots
```

评估结果保存在 `runs/evaluate/<split>/metrics.json`，包含总体指标和逐类 AP。评估其他训练权重：

```bash
python evaluate.py --weights runs/train/pmd_seed0/weights/best.pt --split val --device 0
```

## 检测可视化

```bash
python predict.py --source dataset/val/images/angular_leafspot14.jpg --device 0
python predict.py --source dataset/val/images --device 0
```

支持单张图像和图像目录。默认阈值为 0.2，框线宽度为 3，不显示类别和置信度文字；使用 `--labels` 可显示这些文字。RT-DETR 使用其原有无 NMS 预测流程。结果及预测标注保存到 `runs/predict/`。

```bash
python predict.py --source dataset/val/images/angular_leafspot14.jpg --device 0 --conf 0.2 --line-width 4 --labels
```

## Grad-CAM++

```bash
python visualize.py --source dataset/val/images/angular_leafspot14.jpg --device cuda:0
python visualize.py --source dataset/val/images --device cuda:0 --output runs/gradcam_all
```

默认目标层为 `[16, 20, 23, 26]`，对满足置信度阈值 0.2 的查询类别分数反向传播，采用全图归一化，保留框外响应。添加 `--boxes` 可在热图上叠加预测框。已有同名图片不会被覆盖，可通过 `--output` 指定新目录。

所有入口均提供 `--help`。CPU 运行时，训练、评估和预测使用 `--device cpu`，Grad-CAM++ 同样使用 `--device cpu`。

## 权重校验

随附 `weights/best.pt` 已清理保存于检查点内的旧源码路径和训练目录。全部张量存储数据与训练产出逐字节一致，网络参数与数值未修改；文件校验值随路径元数据的清理而更新。当前 SHA-256 见 `weights/best.pt.sha256`：

```text
f251e01da4f4ed246f53fdbc9334fc24c574ac05633b584a5043479ae2b4486f
```

源码中的普通 Python 注释和说明性文档字符串已清理；第三方版权和许可证声明保留。
