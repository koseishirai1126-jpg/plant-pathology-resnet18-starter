"""保存した最良モデルでテスト画像を予測し、提出用CSVを作る。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PlantTestDataset
from model import get_preprocess, load_trained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/baseline_01"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-sizeには1以上を指定してください。")
    output_path = args.output if args.output is not None else args.run_dir / "submission.csv"
    info_path = output_path.with_suffix(".json")
    if output_path.exists() or info_path.exists():
        parser.error("提出CSVまたは記録ファイルが既にあります。--outputで新しい名前を指定してください。")

    test_df = pd.read_csv(args.data_dir / "test.csv")
    template = pd.read_csv(args.data_dir / "sample_submission.csv")
    if template.columns.tolist() != ["image_id"] + CLASS_NAMES:
        raise ValueError("提出見本の列名・列順が想定と異なります。")
    for name, dataframe in [("test.csv", test_df), ("sample_submission.csv", template)]:
        if dataframe.empty or dataframe["image_id"].isna().any() or dataframe["image_id"].duplicated().any():
            raise ValueError(f"{name}が空か、画像IDに欠損・重複があります。")
    if set(test_df["image_id"]) != set(template["image_id"]):
        raise ValueError("テスト用と提出見本の画像IDが一致しません。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 学習時の検証と同じcuDNNの設定を使う。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_path = args.run_dir / "best.pt"
    model, checkpoint = load_trained_model(checkpoint_path, device)
    transform = get_preprocess()
    dataset = PlantTestDataset(test_df, args.data_dir / "images", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"使用するデバイス: {device_name}", flush=True)
    print(
        f"使用するモデル: {checkpoint['model_name']} epoch={checkpoint['epoch']} "
        f"検証AUC={checkpoint['valid_auc']:.6f}", flush=True,
    )

    image_ids = []
    probability_batches = []
    with torch.no_grad():
        for batch_index, (images, batch_ids) in enumerate(loader, start=1):
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            probability_batches.append(probabilities.cpu().numpy())
            image_ids.extend(batch_ids)
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"  予測バッチ {batch_index}/{len(loader)}", flush=True)

    all_probabilities = np.concatenate(probability_batches, axis=0)
    if image_ids != test_df["image_id"].tolist():
        raise ValueError("予測した画像IDとテスト用CSVの順番が一致しません。")
    prediction_df = pd.DataFrame(all_probabilities, columns=CLASS_NAMES)
    prediction_df.insert(0, "image_id", image_ids)

    # 画像IDで結合し、提出見本の行順と列順に揃える。
    submission = template[["image_id"]].merge(prediction_df, on="image_id", how="left", validate="one_to_one")
    submission = submission[template.columns]
    values = submission[CLASS_NAMES].to_numpy()
    if len(submission) != len(test_df) or submission["image_id"].tolist() != template["image_id"].tolist():
        raise ValueError("提出CSVの画像ID・行数・行順が見本と一致しません。")
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("予測確率に欠損・無限大・範囲外の値があります。")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("画像ごとの予測確率の合計が1になっていません。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, mode="x")
    info = {
        "model": checkpoint["model_name"],
        "pretrained_weights": checkpoint["pretrained_weights"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_valid_auc": checkpoint["valid_auc"],
        "test_csv_sha256": hashlib.sha256((args.data_dir / "test.csv").read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256((args.data_dir / "sample_submission.csv").read_bytes()).hexdigest(),
        "submission_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": len(submission),
        "columns": submission.columns.tolist(),
        "device": device_name,
        "batch_size": args.batch_size,
        "torch": str(torch.__version__),
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(submission.head().to_string(index=False))
    print(f"提出CSVの形: {submission.shape}")
    print("画像ID・列順・確率の形式確認OK")
    print(f"保存先: {output_path.resolve()}")


if __name__ == "__main__":
    main()
