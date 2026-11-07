"""保存した分割から、学習用・検証用のDataLoaderを作る。"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PlantDataset
from model import get_preprocess


def load_split_dataframes(data_dir):
    """split.csvの割り当てを読み込み、正解ラベルを画像IDで対応させる。"""
    data_dir = Path(data_dir)
    all_df = pd.read_csv(data_dir / "train.csv")
    split_df = pd.read_csv(data_dir / "split.csv")

    for name, dataframe in [("train.csv", all_df), ("split.csv", split_df)]:
        if dataframe["image_id"].isna().any() or dataframe["image_id"].duplicated().any():
            raise ValueError(f"{name}の画像IDに欠損または重複があります。")
    if set(split_df["image_id"]) != set(all_df["image_id"]):
        raise ValueError("split.csvの画像IDがtrain.csvと一致しません。")
    if not split_df["split"].isin(["train", "valid"]).all():
        raise ValueError("split列はtrainまたはvalidにしてください。")

    # 行番号ではなく、画像IDを使って正解ラベルと対応させる。
    merged_df = split_df.merge(all_df, on="image_id", validate="one_to_one")
    train_df = merged_df.loc[merged_df["split"] == "train"].reset_index(drop=True)
    valid_df = merged_df.loc[merged_df["split"] == "valid"].reset_index(drop=True)

    for name, dataframe in [("学習用", train_df), ("検証用", valid_df)]:
        if dataframe.empty or (dataframe[CLASS_NAMES].sum() == 0).any():
            raise ValueError(f"{name}には4クラスすべての画像が必要です。")

    return train_df, valid_df


def make_dataloaders(data_dir, batch_size=32, seed=42):
    """同じ分割と前処理を使い、学習用だけ取り出す順番をシャッフルする。"""
    data_dir = Path(data_dir)
    train_df, valid_df = load_split_dataframes(data_dir)

    # 最初のベースラインは、両方とも同じ固定の前処理を使う。
    transform = get_preprocess()
    train_dataset = PlantDataset(train_df, data_dir / "images", transform=transform)
    valid_dataset = PlantDataset(valid_df, data_dir / "images", transform=transform)

    # 学習画像を取り出す順番の乱数を、モデルの初期化とは別に管理する。
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, valid_loader
