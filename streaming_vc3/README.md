# StreamingVC-3

## 1 データセットの準備

`voice_data`に変換先の声を入れる。


## 2 モデルの学習

```
poetry run python bin/train.py --voice_data_dir "voice_data" --testdata_dir "test_data" --asr_ckpt_path "models/asr-latest.pt" --vocoder_ckpt_path "models/hifi-gan-latest.pt"
```