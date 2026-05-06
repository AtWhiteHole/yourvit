"""
下载预训练的教师模型权重
"""

import os
import argparse
import torch
import timm


def download_regnet_teacher(output_dir='./pretrained'):
    """
    下载RegNetY-16GF教师模型权重

    Args:
        output_dir: 保存权重的目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading RegNetY-16GF pretrained weights...")
    print("This may take a few minutes depending on your network speed.")

    # 使用timm下载预训练模型
    model = timm.create_model('regnety_016', pretrained=True)

    # 保存权重
    output_path = os.path.join(output_dir, 'regnetY_16gf_imagenet.pth')
    torch.save(model.state_dict(), output_path)

    print(f"✓ RegNetY-16GF weights saved to: {output_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return output_path


def download_deit_student(output_dir='./pretrained'):
    """
    下载DeiT-Small预训练权重（可选，用于初始化）

    Args:
        output_dir: 保存权重的目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading DeiT-Small pretrained weights...")

    # 使用timm下载预训练模型
    model = timm.create_model('deit_small_patch16_224', pretrained=True)

    # 保存权重
    output_path = os.path.join(output_dir, 'deit_small_patch16_224.pth')
    torch.save(model.state_dict(), output_path)

    print(f"✓ DeiT-Small weights saved to: {output_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download pretrained teacher model weights')
    parser.add_argument('--model', type=str, default='regnetY_16gf',
                        choices=['regnetY_16gf', 'deit_small', 'all'],
                        help='Which model to download')
    parser.add_argument('--output', type=str, default='./pretrained',
                        help='Output directory for weights')

    args = parser.parse_args()

    print("=" * 60)
    print("Downloading Pretrained Model Weights")
    print("=" * 60)

    if args.model == 'regnetY_16gf' or args.model == 'all':
        download_regnet_teacher(args.output)
        print()

    if args.model == 'deit_small' or args.model == 'all':
        download_deit_student(args.output)
        print()

    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nWeights saved in: {args.output}")
    print("\nYou can now run training with:")
    print("  python main.py --teacher-path /root/autodl-tmp/regnetY_16gf_imagenet.pth ...")


if __name__ == '__main__':
    main()
