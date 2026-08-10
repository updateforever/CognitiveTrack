# 独立环境指引

面向 L40 训练服务器的完整自动部署、数据重建、NCCL 和训练验收流程见
[L40 训练服务器部署与 Stage-1 复现](l40_setup.md)。该文档与
`scripts/setup_env.sh` 是当前推荐入口；下面的 `environment.yml` 只提供通用最小
环境，不承担 CUDA/flash-attn 的精确 ABI 复现。

## 1. 全新建立

```bash
cd CognitiveTrack
conda env create -f environment.yml
conda activate cogtrack
pip install -e . --no-build-isolation
```

`environment.yml` 只创建 `cogtrack`，不会修改其他 Conda 环境。如需精确固定
CUDA/PyTorch 组合，建议在集群镜像中先安装与驱动匹配的 PyTorch，再用
`pip install -e . --no-deps --no-build-isolation` 安装本项目。

## 2. 无网络节点

若已有一个通过 Qwen-VL 验证的 Python 3.10 环境，可先克隆再安装本项目：

```bash
conda create --name cogtrack --clone <source-env>
conda run -n cogtrack python -m pip install -e . --no-deps --no-build-isolation
```

不要在公用 baseline 环境中直接升级 `transformers`、`torch` 或 `ms-swift`。

## 3. 本机路径

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

`env.local.yaml` 已被 Git 忽略。可用下列环境变量覆盖 YAML：

- `COGTRACK_MODEL_ROOT`
- `COGTRACK_DATASET_ROOT`
- `COGTRACK_COGNITIVEBENCH_ROOT`
- `COGTRACK_LASOT_ROOT`
- `COGTRACK_TNL2K_ROOT`
- `COGTRACK_MGIT_ROOT`
- `COGTRACK_OUTPUT_ROOT`

模型 YAML 只写 checkpoint 目录名，runner 会用 `model_root` 解析，因此可提交配置
不包开发机绝对路径。

## 4. 自检

```bash
conda run -n cogtrack python -c \
  "import torch, transformers, swift; print(torch.__version__, transformers.__version__, swift.__version__)"
conda run -n cogtrack python tracking/inspect_dataset.py \
  --dataset cognitivebench --config configs/env.local.yaml --limit 1
```

Qwen 推理和 ms-swift 训练建议分开进程启动，训练脚本本身就遵循此边界。

## 5. SUTrack checkpoint

SUTrack 权重不写入 YAML，统一通过环境变量提供：

```bash
export COGTRACK_SUTRACK_CHECKPOINT=/path/to/SUTRACK_ep0180.pth.tar
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/sutrack_b384.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 2
```

内置 B384 runtime 需要 `torch`、`torchvision` 和 `timm`，可用
`pip install -e '.[sutrack]'` 安装。manifest 会记录网络配置和 checkpoint 的
路径、大小与 SHA-256，便于识别权重不一致。
