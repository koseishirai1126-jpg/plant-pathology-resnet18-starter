"""保存した最良モデルを読み直し、検証時の予測を再現できるか確認する。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data_loading import make_dataloaders
from dataset import CLASS_NAMES
from model import load_trained_model
from train import set_seed, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/baseline_01"))
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    for filename, hash_key in [("split.csv", "split_sha256"), ("train.csv", "train_csv_sha256")]:
        current_hash = hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest()
        if current_hash != config[hash_key]:
            raise ValueError(f"{filename}が学習時と異なります。学習時のデータを使ってください。")
    saved_split_hash = hashlib.sha256((args.run_dir / "split.csv").read_bytes()).hexdigest()
    if saved_split_hash != config["split_sha256"]:
        raise ValueError("実行フォルダに保存されたsplit.csvが学習時と異なります。")

    set_seed(config["seed"])
    _, valid_loader = make_dataloaders(
        args.data_dir, batch_size=config["batch_size"], seed=config["seed"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.run_dir / "best.pt"
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if checkpoint["model_name"] != config["model"]:
        raise ValueError("config.jsonとbest.ptのモデル名が一致しません。")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"使用するデバイス: {device_name}", flush=True)
    print(f"モデル: {checkpoint['model_name']}", flush=True)
    print(f"保存されたエポック: {checkpoint['epoch']}", flush=True)

    valid_loss, valid_auc, class_aucs, probabilities = validate(
        model, valid_loader, nn.CrossEntropyLoss(), device,
    )
    saved_predictions = pd.read_csv(args.run_dir / "valid_predictions.csv")
    expected_ids = valid_loader.dataset.dataframe["image_id"].tolist()
    if saved_predictions["image_id"].tolist() != expected_ids:
        raise ValueError("保存された検証予測の画像IDや行順が一致しません。")
    saved_probabilities = saved_predictions[CLASS_NAMES].to_numpy()
    if saved_probabilities.shape != probabilities.shape:
        raise ValueError("保存された検証予測の形が一致しません。")
    if not np.allclose(saved_probabilities, probabilities, rtol=1e-4, atol=1e-6):
        raise ValueError("再読み込み後の予測が保存時と一致しません。データ・前処理・環境を確認してください。")

    report = {
        "model": checkpoint["model_name"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "epoch": checkpoint["epoch"],
        "saved_valid_auc": checkpoint["valid_auc"],
        "reloaded_valid_auc": valid_auc,
        "valid_loss": valid_loss,
        "class_aucs": class_aucs,
        "valid_images": len(expected_ids),
        "maximum_probability_difference": float(np.max(np.abs(saved_probabilities - probabilities))),
        "device": device_name,
    }
    (args.run_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"検証画像の枚数: {len(expected_ids)}")
    print(f"保存時の検証AUC: {checkpoint['valid_auc']:.6f}")
    print(f"再読み込み後の検証AUC: {valid_auc:.6f}")
    print(f"予測確率の最大差: {report['maximum_probability_difference']:.3g}")
    print("保存モデルの再読み込み・検証予測の一致確認OK")


if __name__ == "__main__":
    main()
