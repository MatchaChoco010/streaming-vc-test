# Streaming ASR

## 1 データセットの前処理

学習用wavファイルを入れたdatasetディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/prepare_dataset.py --dataset_dir "dataset/methane_voice_data/emotion/normal" --output_dir "dataset_resampled"
```

## 2 モデルの学習

前処理したdatasetディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/train.py --dataset_dir "dataset_resampled" --testdata_dir "test_data" --feature_extractor_onnx_path "models/feature_extractor.onnx" --encoder_onnx_path "models/encoder.onnx" --vocoder_ckpt_path "models/hifi-gan-best.pt"
```

## 3 hifi-ganのファインチューニング

```
poetry run python bin/finetune.py --dataset_dir "dataset_resampled" --testdata_dir "test_data" --feature_extractor_onnx_path "models/feature_extractor.onnx" --encoder_onnx_path "models/encoder.onnx" --vocoder_ckpt_path "models/hifi-gan-best.pt" --vc_ckpt_path "output/vc/ckpt/exp-20230117-120901/ckpt-latest.pt"
```

## 4 ボイスモデルのパック

次のコマンドでボイスモデルをパックする

```
poetry run python bin/pack_voice_model.py --finetune-ckpt "output/finetune/ckpt/exp-XXXXXXXXXX/ckpt-latest.pt" --cover-img "asset/cover.png" --main-color "#ff5533" --out "./voice.model"
```
