"""ResNet18の学習・検証を行い、検証AUCが最も高いモデルを保存する。"""

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch import nn

from data_loading import make_dataloaders
from dataset import CLASS_NAMES
from model import MODEL_NAME, WEIGHTS, build_model


def set_seed(seed):
    """モデルの初期化などに使う乱数を固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """学習用画像を1周し、パラメータを更新する。"""
    model.train()
    total_loss = 0.0
    total_images = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()          # 前のバッチの勾配を消す
        logits = model(images)         # 予測値を計算する
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("学習時の損失がNaNまたは無限大になりました。")
        loss.backward()                # 勾配を計算する
        optimizer.step()               # パラメータを更新する

        # 最後のバッチの枚数が少なくても、画像1枚あたりの平均になるようにする。
        batch_images = images.size(0)
        total_loss += loss.item() * batch_images
        total_images += batch_images
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"  学習バッチ {batch_index}/{len(loader)} "
                f"平均損失={total_loss / total_images:.4f}",
                flush=True,
            )

    return total_loss / total_images


def validate(model, loader, criterion, device):
    """検証用画像をすべて予測し、損失とクラスごとのROC AUCを求める。"""
    model.eval()
    total_loss = 0.0
    total_images = 0
    probability_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("検証時の損失がNaNまたは無限大になりました。")

            batch_images = images.size(0)
            total_loss += loss.item() * batch_images
            total_images += batch_images
            probabilities = torch.softmax(logits, dim=1)
            probability_batches.append(probabilities.cpu().numpy())
            label_batches.append(labels.cpu().numpy())

    # AUCはバッチごとに計算せず、検証用の全画像をまとめて計算する。
    all_probabilities = np.concatenate(probability_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)
    class_aucs = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_labels = (all_labels == class_index).astype(int)
        class_aucs[class_name] = float(
            roc_auc_score(binary_labels, all_probabilities[:, class_index])
        )
    mean_auc = float(np.mean(list(class_aucs.values())))
    if not math.isfinite(mean_auc):
        raise ValueError("検証AUCが有限の数値になりませんでした。")

    return total_loss / total_images, mean_auc, class_aucs, all_probabilities


def get_code_version():
    """実行したコードのGitコミットと、未コミットの変更の有無を調べる。"""
    project_dir = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir,
            capture_output=True, text=True, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_dir,
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_01"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochsとbatch-sizeには1以上を指定してください。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lrには0より大きい有限の値を指定してください。")
    if args.output_dir.exists():
        parser.error("output-dirは未作成のフォルダを指定してください。既存の結果は上書きしません。")

    set_seed(args.seed)
    train_loader, valid_loader = make_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"モデル: {MODEL_NAME}", flush=True)
    print(f"使用するデバイス: {device_name}", flush=True)
    print(
        f"学習用: {len(train_loader.dataset)}枚 / "
        f"検証用: {len(valid_loader.dataset)}枚",
        flush=True,
    )

    # 全パラメータを更新する。毎回ImageNetの重みから新しく学習を始める。
    model = build_model(pretrained=True).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"パラメータ数（4クラス用）: {parameter_count:,}", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_path = args.data_dir / "split.csv"
    shutil.copyfile(split_path, args.output_dir / "split.csv")
    config = {
        "model": MODEL_NAME,
        "pretrained_weights": WEIGHTS.name,
        "parameter_count": parameter_count,
        "class_names": CLASS_NAMES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "seed": args.seed,
        "device": device_name,
        "train_images": len(train_loader.dataset),
        "valid_images": len(valid_loader.dataset),
        "preprocessing": "Resize(256), CenterCrop(224), ToTensor, ImageNet normalization",
        "augmentation": None,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "train_csv_sha256": hashlib.sha256((args.data_dir / "train.csv").read_bytes()).hexdigest(),
        "code": get_code_version(),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    best_auc = -math.inf
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_auc, class_aucs, probabilities = validate(
            model, valid_loader, criterion, device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_auc": valid_auc,
            **{f"auc_{name}": value for name, value in class_aucs.items()},
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_metrics)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(
            f"  train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_auc={valid_auc:.6f}",
            flush=True,
        )
        print("  " + " / ".join(f"{name}={auc:.6f}" for name, auc in class_aucs.items()), flush=True)

        # 平均AUCが改善した時点の重みと、その重みでの検証予測を保存する。
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "pretrained_weights": WEIGHTS.name,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "valid_auc": valid_auc,
                },
                args.output_dir / "best.pt",
            )
            prediction_df = pd.DataFrame(probabilities, columns=CLASS_NAMES)
            prediction_df.insert(0, "image_id", valid_loader.dataset.dataframe["image_id"].to_numpy())
            prediction_df.to_csv(args.output_dir / "valid_predictions.csv", index=False)
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(epoch_metrics, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print("  最良モデルを保存しました。", flush=True)

    print(f"学習完了: best_epoch={best_epoch} best_valid_auc={best_auc:.6f}")
    print(f"保存先: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
