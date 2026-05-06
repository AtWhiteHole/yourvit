# Token Fusion with Knowledge Distillation for Vision Transformer

基于Token融合与知识蒸馏的轻量化Vision Transformer实现

## 项目简介

本项目实现了一种高效的Vision Transformer加速方法，通过Token融合技术降低计算复杂度，并结合知识蒸馏补偿精度损失。

**核心特性**：
- Token融合：在指定层动态选择显著token，将非显著token加权融合
- 渐进式收缩：训练过程中平滑降低token保留率
- 自适应蒸馏：蒸馏权重随Token压缩强度同步调整
- 高效推理：在保持精度的同时显著降低计算量和推理延迟

**实验结果**（ImageNet-1K子集）：
- 计算量降低：4.6G → 3.0G FLOPs（降幅34.8%）
- 推理加速：6.2ms → 4.1ms（提速34%）
- 精度保持：通过自适应蒸馏，Top-1准确率达到75.7%

---

## 快速开始

### 1. 环境配置

```bash
# 安装PyTorch
pip install torch==1.9.0 torchvision==0.10.0

# 安装其他依赖
pip install timm==0.4.12 tensorboardX==2.4 torchprofile==0.0.4 lmdb==1.2.1 pyarrow==5.0.0 einops==0.4.1
```

### 2. 准备数据集

将ImageNet子集放在某个路径，目录结构：
```
/path/to/imagenet_subset/
├── train/
│   ├── n01440764/
│   │   ├── image1.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### 3. 下载预训练权重

```bash
python download_teacher.py --model regnetY_16gf --output ./pretrained
```

### 4. 运行训练

#### 最小可行实验集（3个核心实验）

**实验1：基线模型**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /path/to/imagenet_subset \
  --output_dir ./outputs/exp1_baseline \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 1.0 \
  --distillation-type none
```

**实验4：Token融合（r=0.7）**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /path/to/imagenet_subset \
  --output_dir ./outputs/exp4_fusion_r07 \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

**实验11：完整方法（Token融合 + 自适应蒸馏）**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --teacher-model regnetY_16gf \
  --teacher-path /root/autodl-tmp/regnetY_16gf_imagenet.pth \
  --data-path /path/to/imagenet_subset \
  --output_dir ./outputs/exp11_full_method \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type adaptive_soft_hard \
  --distillation-alpha-start 0.1 \
  --distillation-alpha-end 0.5 \
  --distillation-tau 4.0
```

### 5. 生成论文图表

**训练曲线图**：
```bash
python plot_training_curves.py \
  --log-dir ./outputs/exp11_full_method \
  --output ./figures/fig4_1_training_curves.png
```

**Token可视化图**：
```bash
CUDA_VISIBLE_DEVICES=0 python visualize_mask.py \
  --model deit_small_patch16_224 \
  --data-path /path/to/imagenet_subset \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp11_full_method/checkpoint.pth \
  --num-vis 10 \
  --output-dir ./figures/token_visualization
```

---

## 项目结构

```
vit-token-fusion-kd/
├── README.md                      # 本文件
├── EXPERIMENTS.md                 # 所有实验的详细命令
├── requirements.txt               # Python依赖
├── main.py                        # 主训练脚本
├── evit.py                        # 模型定义（含Token融合）
├── losses.py                      # 损失函数（含知识蒸馏）
├── engine.py                      # 训练和评估逻辑
├── datasets.py                    # 数据加载
├── models.py                      # 模型创建
├── download_teacher.py            # 下载教师模型权重
├── plot_training_curves.py        # 绘制训练曲线
├── visualize_mask.py              # Token融合可视化
├── speed_test.py                  # 推理速度测试
└── utils.py                       # 工具函数
```

---

## 实验列表

详见 `EXPERIMENTS.md` 文件，包含：

- **表4.3实验**：不同Token保留率（r=1.0, 0.9, 0.8, 0.7, 0.6, 0.5）
- **表4.4实验**：不同蒸馏策略（无蒸馏、软蒸馏、硬蒸馏、固定联合、自适应联合）
- **图表生成**：训练曲线和Token可视化

---

## 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 学生模型 | deit_small_patch16_224 |
| `--teacher-model` | 教师模型 | regnetY_16gf |
| `--teacher-path` | 教师模型权重路径 | - |
| `--base_keep_rate` | Token保留率 | 0.7 |
| `--fuse_token` | 是否融合非显著token | False |
| `--drop_loc` | 融合层位置 | "(3, 6, 9)" |
| `--shrink_start_epoch` | 开始收缩的epoch | 30 |
| `--shrink_epochs` | 收缩持续的epoch数 | 120 |
| `--distillation-type` | 蒸馏类型 | none |
| `--distillation-alpha-start` | 蒸馏权重起始值 | 0.1 |
| `--distillation-alpha-end` | 蒸馏权重结束值 | 0.5 |
| `--distillation-tau` | 蒸馏温度 | 4.0 |
| `--epochs` | 训练轮数 | 300 |
| `--batch-size` | 批次大小 | 256 |
| `--lr` | 学习率 | 1e-3 |

---

## 多服务器并行训练

如果有多台服务器，可以并行运行不同实验：

**服务器1**：
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install -r requirements.txt
export DATA_PATH=/path/to/imagenet_subset
CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --data-path $DATA_PATH --output_dir ./outputs/exp1_baseline --batch-size 256 --epochs 300 --base_keep_rate 1.0 --distillation-type none
```

**服务器2**：
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install -r requirements.txt
export DATA_PATH=/path/to/imagenet_subset
CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --data-path $DATA_PATH --output_dir ./outputs/exp4_fusion_r07 --batch-size 256 --epochs 300 --base_keep_rate 0.7 --fuse_token --drop_loc "(3, 6, 9)" --shrink_start_epoch 30 --shrink_epochs 120 --distillation-type none
```

**服务器3**：
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install -r requirements.txt
python download_teacher.py
export DATA_PATH=/path/to/imagenet_subset
CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --teacher-model regnetY_16gf --teacher-path /root/autodl-tmp/regnetY_16gf_imagenet.pth --data-path $DATA_PATH --output_dir ./outputs/exp11_full_method --batch-size 256 --epochs 300 --base_keep_rate 0.7 --fuse_token --drop_loc "(3, 6, 9)" --shrink_start_epoch 30 --shrink_epochs 120 --distillation-type adaptive_soft_hard --distillation-alpha-start 0.1 --distillation-alpha-end 0.5 --distillation-tau 4.0
```

---

## 常见问题

### Q1: CUDA out of memory
**解决方案**：减小batch size
```bash
--batch-size 128  # 或 64
```

### Q2: 训练速度慢
**解决方案**：
1. 减少验证频率
2. 使用更少的workers
3. 快速验证时减少epochs：`--epochs 100`

### Q3: 如何恢复训练
```bash
--resume ./outputs/exp1_baseline/checkpoint.pth
```

### Q4: 如何只测试不训练
```bash
--eval --resume ./outputs/exp1_baseline/checkpoint.pth
```

### Q5: 如何测量推理速度
```bash
--test_speed --eval --resume ./outputs/exp1_baseline/checkpoint.pth
```

---

## 论文相关文档

- `thesis-modifications.md` - 论文修改清单（修改内容和位置）
- `experiment-checklist.md` - 实验完成情况清单
- `figure-generation-guide.md` - 图表生成指南

---

## 引用

如果使用本代码，请引用：

```bibtex
@article{zhang2026token,
  title={基于蒸馏与Token融合协同优化的轻量化Vision Transformer设计与实现},
  author={张心雅},
  journal={华南理工大学本科毕业论文},
  year={2026}
}
```

---

## License

MIT License

---

## 联系方式

如有问题，请提交Issue或联系作者。
