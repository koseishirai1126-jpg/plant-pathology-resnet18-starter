# Plant Pathology 2020 実践演習

ImageNetで学習済みのResNet18を使い、リンゴの葉の画像を4クラスに分類します。基本モデルを動かし、学習・検証・Kaggle提出までを一度通してから、各自の仮説を試します。

![葉の画像から4クラスを予測する流れ](docs/resnet18.png)

| 比較用モデル | Kaggle Private | Kaggle Public |
| --- | ---: | ---: |
| 標準前処理＋ResNet18 | 0.93658 | 0.94415 |
| 追加の画像処理＋ResNet18 | **0.95140** | 0.95217 |

演習の目標は、Kaggleの **Privateスコア0.95140を超えること**です。共通の検証分割で改善案を比較し、採用する条件を決めてから提出します。

## 準備

PyTorchとtorchvisionが動く計算機のPython環境を使います。研究室で基本モデルを学習した環境はPython 3.13.11、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128です。初回の学習ではImageNetの重みをダウンロードします。

残りの依存パッケージが未導入なら、使用中のPython環境でインストールします。

```bash
python -m pip install -r requirements.txt
```

各自でデータを取得して展開し、リポジトリのルートに次の構成で置きます。[コンペのデータ](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data)

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  images/
```

配布された共通の `split.csv` を `data/split.csv` に置きます。手元で動作を試すために分割を作る場合は `python prepare_split.py` を使えます。比較するときは同じ画像IDの割り当てを使います。

## 学習して提出CSVを作る

```bash
python train.py --output-dir outputs/baseline_01
python evaluate.py --run-dir outputs/baseline_01
python predict.py --run-dir outputs/baseline_01
```

初期設定は5エポック、バッチサイズ32、学習率0.0001、乱数シード42です。`train.py` は検証AUCが最も高かった重みを保存します。`evaluate.py` はその重みを読み直して検証予測を照合し、`predict.py` はテスト画像の4クラスの予測確率を `outputs/baseline_01/submission.csv` に保存します。再実行するときは新しい出力名を使います。

## 学習ガイド

参考書の章に合わせて実際のコードを読み、小さい確認から順に進めます。[PyTorch学習ガイド（仮版）](docs/learning-guide.md)
