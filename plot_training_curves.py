"""
绘制训练曲线图
从TensorBoard日志中提取训练数据并生成可视化图表
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_tensorboard_data(log_dir):
    """
    从TensorBoard日志文件中加载数据

    Args:
        log_dir: TensorBoard日志目录

    Returns:
        dict: 包含训练数据的字典
    """
    # 查找events文件
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    # 加载最新的events文件
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Loading data from: {event_file}")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # 提取数据
    data = {}

    # 获取所有可用的标量
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data


def plot_training_curves(data, output_path, title="训练过程曲线"):
    """
    绘制训练曲线

    Args:
        data: 训练数据字典
        output_path: 输出图像路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1: 损失曲线
    ax1 = axes[0]

    # 绘制训练损失
    if 'train/loss' in data:
        ax1.plot(data['train/loss']['steps'], data['train/loss']['values'],
                label='训练损失', linewidth=2, alpha=0.8)
    elif 'loss' in data:
        ax1.plot(data['loss']['steps'], data['loss']['values'],
                label='训练损失', linewidth=2, alpha=0.8)

    # 绘制验证损失
    if 'val/loss' in data:
        ax1.plot(data['val/loss']['steps'], data['val/loss']['values'],
                label='验证损失', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 子图2: 准确率曲线
    ax2 = axes[1]

    # 绘制Top-1准确率
    if 'train/acc1' in data:
        ax2.plot(data['train/acc1']['steps'], data['train/acc1']['values'],
                label='训练Top-1准确率', linewidth=2, alpha=0.8)
    elif 'acc1' in data:
        ax2.plot(data['acc1']['steps'], data['acc1']['values'],
                label='训练Top-1准确率', linewidth=2, alpha=0.8)

    # 绘制验证Top-1准确率
    if 'val/acc1' in data:
        ax2.plot(data['val/acc1']['steps'], data['val/acc1']['values'],
                label='验证Top-1准确率', linewidth=2, alpha=0.8, color='orange')
    elif 'test_acc1' in data:
        ax2.plot(data['test_acc1']['steps'], data['test_acc1']['values'],
                label='验证Top-1准确率', linewidth=2, alpha=0.8, color='orange')

    # 绘制Top-5准确率（如果有）
    if 'val/acc5' in data:
        ax2.plot(data['val/acc5']['steps'], data['val/acc5']['values'],
                label='验证Top-5准确率', linewidth=2, alpha=0.8, linestyle='--')
    elif 'test_acc5' in data:
        ax2.plot(data['test_acc5']['steps'], data['test_acc5']['values'],
                label='验证Top-5准确率', linewidth=2, alpha=0.8, linestyle='--')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default='./training_curves.png',
                        help='Output image path')
    parser.add_argument('--title', type=str, default='训练过程曲线',
                        help='Plot title')

    args = parser.parse_args()

    # 加载数据
    print("Loading TensorBoard data...")
    data = load_tensorboard_data(args.log_dir)

    # 绘制曲线
    print("Plotting training curves...")
    plot_training_curves(data, args.output, args.title)

    print("Done!")


if __name__ == '__main__':
    main()
