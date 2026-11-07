"""実データの先頭3枚でDatasetの動作を確認する。学習は行わない。"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision.models import ResNet18_Weights

from dataset import CLASS_NAMES, PlantDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")

    # 学習済みResNet18に対応する前処理。ここでは重みのダウンロードは行わない。
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    dataset = PlantDataset(train_df, args.data_dir / "images", transform=transform)

    print(f"画像の枚数: {len(dataset)}")
    for index in range(min(3, len(dataset))):
        image, label = dataset[index]
        image_id = train_df.iloc[index]["image_id"]

        if image.shape != (3, 224, 224) or image.dtype != torch.float32:
            raise ValueError(f"{image_id}: 画像の形または型が想定と異なります。")
        if not torch.isfinite(image).all():
            raise ValueError(f"{image_id}: 画像にNaNまたは無限大が含まれています。")

        print(
            f"{image_id}: shape={tuple(image.shape)}, dtype={image.dtype}, "
            f"label={label} ({CLASS_NAMES[label]})"
        )

    print("Datasetの動作確認OK")


if __name__ == "__main__":
    main()
