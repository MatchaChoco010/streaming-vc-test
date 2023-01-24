# Streaming ASR

## 1 Huggingfaceにログインする

```
poetry run huggingface-cli login
```

## 2 モデルの学習

```
poetry run python bin/train.py
```

## 3 onnxモデルの出力

```
poetry run python bin/export.py --ckpt "output/ckpt/exp-20230123-054937/ckpt-latest.pt"
```
