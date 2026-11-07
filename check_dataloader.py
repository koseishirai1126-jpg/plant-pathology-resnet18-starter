"""Datasetの画像と正解ラベルを4枚ずつまとめて取り出す。"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import CLASS_NAMES, PlantDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    dataset = PlantDataset(train_df, args.data_dir / "images", transform=transform)

    # 今回はCSVの順に4枚ずつ取り出し、最後の端数も残す。
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # 先頭の1バッチを取得する。この段階ではモデルの学習は行わない。
    images, labels = next(iter(loader))

    if images.shape != (len(labels), 3, 224, 224):
        raise ValueError("画像のバッチの形が想定と異なります。")
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("正解ラベルのバッチの形または型が想定と異なります。")

    print(f"画像の枚数: {len(dataset)}")
    print(f"バッチの数: {len(loader)}")
    print(f"画像の形: {tuple(images.shape)}")
    print(f"画像の型: {images.dtype}")
    print(f"ラベルの形: {tuple(labels.shape)}")
    print(f"ラベルの型: {labels.dtype}")
    print(f"正解ラベル: {labels.tolist()}")
    print(f"対応するクラス: {[CLASS_NAMES[label] for label in labels.tolist()]}")
    print("DataLoaderの動作確認OK")


if __name__ == "__main__":
    main()
