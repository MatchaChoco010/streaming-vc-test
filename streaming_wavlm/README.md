# Streaming WavLM

## 1 k-meansモデルを作る

```
poetry run python bin/fit_km.py
```

## 2 WavLMの訓練を開始する


```
poetry run python bin/train.py --km_path="models/km_model.km"
```
