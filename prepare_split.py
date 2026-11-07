"""train.csvをクラスの比率を保って学習用80%・検証用20%に分割する。"""

import argparse
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from dataset import CLASS_NAMES


VALID_SIZE = 0.2
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    all_df = pd.read_csv(args.data_dir / "train.csv")
    if all_df["image_id"].isna().any() or all_df["image_id"].duplicated().any():
        raise ValueError("画像IDに欠損または重複があります。")

    # CSVの行順が変わっても、同じ並びから分割できるようIDで並べる。
    all_df = all_df.sort_values("image_id").reset_index(drop=True)
    one_hot_labels = all_df[CLASS_NAMES].to_numpy()
    if not ((one_hot_labels == 0) | (one_hot_labels == 1)).all():
        raise ValueError("正解ラベルには0または1を指定してください。")
    if not (one_hot_labels.sum(axis=1) == 1).all():
        raise ValueError("正解ラベルは各画像につき1クラスにしてください。")
    labels = one_hot_labels.argmax(axis=1)

    train_df, valid_df = train_test_split(
        all_df,
        test_size=VALID_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_ids = set(train_df["image_id"])
    valid_ids = set(valid_df["image_id"])
    if train_ids & valid_ids:
        raise ValueError("学習用と検証用に同じ画像が含まれています。")
    if train_ids | valid_ids != set(all_df["image_id"]):
        raise ValueError("分割に含まれていない画像があります。")

    # 元画像やラベルはコピーせず、画像IDと割り当てだけを保存する。
    split_df = all_df[["image_id"]].copy()
    split_df["split"] = "train"
    split_df.loc[split_df["image_id"].isin(valid_ids), "split"] = "valid"
    split_path = args.data_dir / "split.csv"

    # 既存の分割が異なる場合は、比較条件を勝手に変えずに止める。
    if split_path.exists():
        saved_split = pd.read_csv(split_path)
        if not saved_split.equals(split_df):
            raise ValueError("既存のsplit.csvと分割が異なります。上書きせず終了します。")
        print(f"同じ分割を確認: {split_path}")
    else:
        split_df.to_csv(split_path, index=False, mode="x")
        print(f"分割を保存: {split_path}")

    counts = pd.DataFrame(
        {
            "全体": all_df[CLASS_NAMES].sum(),
            "学習用": train_df[CLASS_NAMES].sum(),
            "検証用": valid_df[CLASS_NAMES].sum(),
        }
    )
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"乱数の設定: {RANDOM_SEED}")
    print(f"学習用: {len(train_df)}枚")
    print(f"検証用: {len(valid_df)}枚")
    print(counts)
    print("学習用・検証用の画像IDの重複: 0件")


if __name__ == "__main__":
    main()
