# Streaming ASR

## 1 データセットの準備と前処理

学習用wavファイルを入れたvoice_dataディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/prepare_voice_data.py --voice_data_dir "voice_data/methane_voice_data/emotion/normal" --output_dir "voice_data_resampled"
```

## 2 モデルの学習

前処理したvoice_data_resampledディレクトリを渡して次のコマンドを実行する。

```
poetry run python bin/vc_train.py --voice_data_dir "voice_data_resampled" --testdata_dir "test_data" --asr_ckpt_path "models/asr-best.pt" --vocoder_ckpt_path "models/hifi-gan-best.pt"
```

## 3 hifi-ganのファインチューニング

```
poetry run python bin/finetune.py --voice_data_dir "voice_data_resampled" --testdata_dir "test_data" --feature_extractor_onnx_path "models/feature_extractor.onnx" --encoder_onnx_path "models/encoder.onnx" --vocoder_ckpt_path "models/hifi-gan-best.pt" --vc_ckpt_path "output/vc/ckpt/exp-20230124-175320/ckpt-latest.pt"
```

## 4 ボイスモデルのパック

次のコマンドでボイスモデルをパックする

```
poetry run python bin/pack_voice_model.py --finetune-ckpt "output/finetune/ckpt/exp-XXXXXXXXXX/ckpt-latest.pt" --cover-img "asset/cover.png" --main-color "#ff5533" --out "./voice.model"
```
