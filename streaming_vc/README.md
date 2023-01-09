# Streaming ASR

## 1 データセットの前処理

学習用wavファイルを入れたdatasetディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/prepare_dataset.py --dataset_dir "dataset/methane_voice_data/emotion/normal" --output_dir "dataset_resampled"
```

## 2 モデルの学習

前処理したdatasetディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/train.py --dataset_dir "dataset_resampled" --testdata_dir "test_data" --feature_extractor_onnx_path "models/feature_extractor.onnx" --encoder_onnx_path "models/encoder.onnx" --vocoder_ckpt_path "models/hifi-gan.ckpt"
```

## 3 hifi-ganのファインチューニング

```
poetry run python bin/finetune.py --dataset_dir "dataset_resamplped" --feature_extractor_onnx_path "models/feature_extractor.onnx" --encoder_onnx_path "models/encoder.onnx" --vc-model "output/vc/ckpt/exp-XXXXXXXXXXX/ckpt-latest.ckpt"
```

## 4 ボイスモデルのパック

次のコマンドでボイスモデルをパックする

```
poetry run python bin/pack_voice_model.py --hifi-gan-ckpt "output/hifi-gan/ckpt/exp-XXXXXXXXXX/finietune-latest.ckpt" --vc-model "output/vc/ckpt/exp-XXXXXXXXXXX/ckpt-latest.ckpt" --cover-img "asset/cover.png" --main-color "#ff5533" --out "./voice.model"
```
