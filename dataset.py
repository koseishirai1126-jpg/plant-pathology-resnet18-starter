"""学習・検証用の画像と正解ラベル、またはテスト用の画像とIDを取り出す。"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# クラス番号はこの並び順で固定する。
CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]


class PlantDataset(Dataset):
    """正解ラベルを含むDataFrameと画像フォルダを受け取るDataset。"""

    def __init__(self, dataframe, image_dir, transform=None):
        # 分割後のDataFrameでも、0番目から位置で取り出せるようにする。
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform

        # argmaxで変換する前に、各行が正しいone-hotラベルか確認する。
        one_hot_labels = self.dataframe[CLASS_NAMES].to_numpy()
        if not ((one_hot_labels == 0) | (one_hot_labels == 1)).all():
            raise ValueError("正解ラベルには0または1を指定してください。")
        if not (one_hot_labels.sum(axis=1) == 1).all():
            raise ValueError("正解ラベルは各画像につき1クラスにしてください。")

        # 例: [0, 0, 0, 1] → 3（scab）。列の順番はCLASS_NAMESに合わせる。
        self.labels = one_hot_labels.argmax(axis=1)

    def __len__(self):
        """このDatasetに含まれる画像の枚数を返す。"""
        return len(self.dataframe)

    def __getitem__(self, index):
        """指定した位置の画像と、クラス番号を返す。"""
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / f"{row['image_id']}.jpg"

        # RGBの3チャンネルに揃え、読み込み後に元ファイルを閉じる。
        with Image.open(image_path) as original_image:
            image = original_image.convert("RGB")

        # リサイズやTensorへの変換などを、呼び出し側から指定できる。
        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[index])
        return image, label


class PlantTestDataset(Dataset):
    """正解ラベルのないテスト画像と、その画像IDを返すDataset。"""

    def __init__(self, dataframe, image_dir, transform):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = self.dataframe.iloc[index]["image_id"]
        image_path = self.image_dir / f"{image_id}.jpg"
        with Image.open(image_path) as original_image:
            image = original_image.convert("RGB")
        image = self.transform(image)

        # 正解ラベルの代わりにIDを返し、予測と画像の対応を保つ。
        return image, image_id
