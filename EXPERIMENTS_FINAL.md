# 实验操作步骤（规范版）

## 项目使用的Python脚本

本项目需要运行以下核心脚本：

1. **main.py** - 主训练脚本（必需）
2. **download_teacher.py** - 下载教师模型权重（必需，仅运行一次）
3. **plot_training_curves.py** - 生成训练曲线图（实验后运行）
4. **visualize_mask.py** - 生成Token可视化图（实验后运行）

## 环境配置

### 1. 安装依赖

```bash
pip install torch==1.9.0 torchvision==0.10.0
pip install timm==0.4.12 tensorboardX==2.4 torchprofile==0.0.4 lmdb==1.2.1 pyarrow==5.0.0 einops==0.4.1
```

### 2. 数据集路径

**固定路径**：`/autodo-tmp/imagenet`

数据集目录结构：
```
/autodo-tmp/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── image1.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### 3. 下载教师模型（仅需运行一次）

```bash
cd /path/to/vit-token-fusion-kd
python download_teacher.py --model regnetY_16gf --output ./pretrained
```

---

## 核心实验（最小可行集）

### 实验1：基线模型（r=1.0，无Token融合，无蒸馏）

**对应论文**：表4.3第1行

**命令**：
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp1_baseline \
  --batch-size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 30 \
  --base_keep_rate 1.0 \
  --distillation-type none
```

**预期结果**：
- Top-1准确率：~76.0%
- FLOPs：4.6G
- 推理延迟：~6.2ms

---

### 实验4：Token融合 r=0.7（无蒸馏）

**对应论文**：表4.3第4行，表4.4第1行（无蒸馏基线）

**命令**：
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp4_fusion_r07 \
  --batch-size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 30 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

**预期结果**：
- Top-1准确率：~74.3%
- FLOPs：3.0G
- 推理延迟：~4.1ms

---

### 实验11：完整方法（r=0.7 + 自适应蒸馏）【论文核心结果】

**对应论文**：表4.4第5行

**命令**：
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --teacher-model regnety_160 \
  --teacher-path ./pretrained/regnetY_16gf_imagenet.pth \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp11_full_method \
  --batch-size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 30 \
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

**预期结果**：
- Top-1准确率：~75.7%
- FLOPs：3.0G
- 推理延迟：~4.1ms

---

## 完整实验列表

### 表4.3：不同Token保留率实验

#### 实验2：r=0.9
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp2_fusion_r09 \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.9 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

#### 实验3：r=0.8
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp3_fusion_r08 \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.8 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

#### 实验5：r=0.6
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp5_fusion_r06 \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.6 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

#### 实验6：r=0.5
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp6_fusion_r05 \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.5 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type none
```

---

### 表4.4：不同蒸馏策略实验（r=0.7）

#### 实验8：常规软蒸馏
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --teacher-model regnety_160 \
  --teacher-path ./pretrained/regnetY_16gf_imagenet.pth \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp8_soft_distill \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type soft \
  --distillation-alpha 0.5 \
  --distillation-tau 4.0
```

#### 实验9：硬蒸馏
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --teacher-model regnety_160 \
  --teacher-path ./pretrained/regnetY_16gf_imagenet.pth \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp9_hard_distill \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type hard
```

#### 实验10：固定软硬联合蒸馏
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --teacher-model regnety_160 \
  --teacher-path ./pretrained/regnetY_16gf_imagenet.pth \
  --data-path /autodo-tmp/imagenet \
  --output_dir ./outputs/exp10_fixed_soft_hard \
  --batch-size 256 \
  --epochs 300 \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --shrink_start_epoch 30 \
  --shrink_epochs 120 \
  --distillation-type soft_hard \
  --distillation-alpha 0.5 \
  --distillation-tau 4.0
```

---

## 图表生成

### 生成训练曲线图（图4.1）

训练完成后运行：
```bash
python plot_training_curves.py \
  --log-dir ./outputs/exp11_full_method \
  --output ./figures/fig4_1_training_curves.png \
  --title "完整方法训练过程曲线"
```

### 生成Token可视化图（图3.5）

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_mask.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp11_full_method/checkpoint.pth \
  --n_visualization 10 \
  --visualize_mask \
  --output_dir ./figures/token_visualization
```

---

## 评估命令

训练完成后评估模型：

```bash
# 评估实验1
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --resume ./outputs/exp1_baseline/checkpoint.pth \
  --eval

# 评估实验4
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp4_fusion_r07/checkpoint.pth \
  --eval

# 评估实验11
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp11_full_method/checkpoint.pth \
  --eval
```

---

## 测量推理速度

```bash
# 测量实验1速度
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --resume ./outputs/exp1_baseline/checkpoint.pth \
  --test_speed \
  --eval

# 测量实验4速度
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp4_fusion_r07/checkpoint.pth \
  --test_speed \
  --eval

# 测量实验11速度
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model deit_small_patch16_224 \
  --data-path /autodo-tmp/imagenet \
  --base_keep_rate 0.7 \
  --fuse_token \
  --drop_loc "(3, 6, 9)" \
  --resume ./outputs/exp11_full_method/checkpoint.pth \
  --test_speed \
  --eval
```

---

## 并行运行（多服务器）

### 服务器1：运行实验1
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install torch==1.9.0 torchvision==0.10.0
pip install timm==0.4.12 tensorboardX==2.4 torchprofile==0.0.4 lmdb==1.2.1 pyarrow==5.0.0 einops==0.4.1

CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --data-path /autodo-tmp/imagenet --output_dir ./outputs/exp1_baseline --batch-size 256 --epochs 300 --base_keep_rate 1.0 --distillation-type none
```

### 服务器2：运行实验4
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install torch==1.9.0 torchvision==0.10.0
pip install timm==0.4.12 tensorboardX==2.4 torchprofile==0.0.4 lmdb==1.2.1 pyarrow==5.0.0 einops==0.4.1

CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --data-path /autodo-tmp/imagenet --output_dir ./outputs/exp4_fusion_r07 --batch-size 256 --epochs 300 --base_keep_rate 0.7 --fuse_token --drop_loc "(3, 6, 9)" --shrink_start_epoch 30 --shrink_epochs 120 --distillation-type none
```

### 服务器3：运行实验11
```bash
git clone https://github.com/your-username/vit-token-fusion-kd.git
cd vit-token-fusion-kd
pip install torch==1.9.0 torchvision==0.10.0
pip install timm==0.4.12 tensorboardX==2.4 torchprofile==0.0.4 lmdb==1.2.1 pyarrow==5.0.0 einops==0.4.1

# 下载教师模型
python download_teacher.py --model regnetY_16gf --output ./pretrained

# 运行训练
CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_patch16_224 --teacher-model regnety_160 --teacher-path ./pretrained/regnetY_16gf_imagenet.pth --data-path /autodo-tmp/imagenet --output_dir ./outputs/exp11_full_method --batch-size 256 --epochs 300 --base_keep_rate 0.7 --fuse_token --drop_loc "(3, 6, 9)" --shrink_start_epoch 30 --shrink_epochs 120 --distillation-type adaptive_soft_hard --distillation-alpha-start 0.1 --distillation-alpha-end 0.5 --distillation-tau 4.0
```

---

## 注意事项

1. **数据集路径**：固定为 `/autodo-tmp/imagenet`，无需修改

2. **GPU选择**：
   - 第一块GPU：`CUDA_VISIBLE_DEVICES=0`
   - 第二块GPU：`CUDA_VISIBLE_DEVICES=1`

3. **Batch Size**：如果显存不足，减小batch size：
   ```bash
   --batch-size 128  # 或 64
   ```

4. **快速验证**：减少epochs快速测试：
   ```bash
   --epochs 100
   ```

5. **恢复训练**：
   ```bash
   --resume ./outputs/exp1_baseline/checkpoint.pth
   ```

6. **教师模型**：实验8-11需要RegNetY-16GF权重，运行前确保下载

7. **输出目录**：所有结果保存在 `./outputs/` 下对应的实验文件夹中

8. **图表生成**：训练完成后运行可视化脚本生成论文所需图表

---

## 自适应蒸馏说明

本项目实现的自适应蒸馏策略参考了DeiT和DearKD两篇论文：

- **DeiT (ICML 2021)**：提出了软硬标签联合蒸馏方法
- **DearKD (CVPR 2022)**：提出在训练早期使用更强的蒸馏来提高数据效率

**我们的策略**：
- 阶段1（Warmup）：使用轻度蒸馏（alpha=0.1）
- 阶段2（渐进融合）：随Token数量减少，线性增加蒸馏权重至0.5
- 阶段3（稳定训练）：保持固定蒸馏权重（alpha=0.5）

这种策略使蒸馏强度与Token压缩强度同步变化，形成补偿机制。
