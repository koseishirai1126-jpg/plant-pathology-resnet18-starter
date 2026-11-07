"""葉の画像を4クラスに分類するResNet18。"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from dataset import CLASS_NAMES


MODEL_NAME = "resnet18"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def get_preprocess():
    """学習・検証・テストで共通の標準前処理を返す。"""
    return WEIGHTS.transforms()


def build_model(pretrained=True):
    """ImageNetの学習済みResNet18を、4クラス用に組み替える。"""
    model = resnet18(weights=WEIGHTS if pretrained else None)
    # 最後の分類層を、512個の特徴から4個のlogitを出す層へ置き換える。
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_trained_model(checkpoint_path, device):
    """保存した重みを使って、推論用のモデルを作る。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("保存したモデルとコードでクラスの並び順が異なります。")
    # モデル名を保存していなかった初期のResNet18の重みにも対応する。
    if checkpoint.get("model_name", MODEL_NAME) != MODEL_NAME:
        raise ValueError("ResNet18の重みを指定してください。")
    if checkpoint.get("pretrained_weights", WEIGHTS.name) != WEIGHTS.name:
        raise ValueError("保存した重みとコードで、学習済み重みの設定が異なります。")
    checkpoint.setdefault("model_name", MODEL_NAME)
    checkpoint.setdefault("pretrained_weights", WEIGHTS.name)

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
