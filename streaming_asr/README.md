# Streaming ASR

## 1 データセットのダウンロードと前処理

```
poetry run python bin/download_dataset.py
poetry run python bin/prepare_dataset.py
```

## 2 モデルの学習

```
poetry run python bin/train.py
```

## 3 onnxモデルの出力

```
poetry run python bin/export.py --ckpt "output/ckpt/exp-20230104-083821/ckpt-00220000.pt"
```
