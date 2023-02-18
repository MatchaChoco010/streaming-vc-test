# Streaming VC

## 1 データセットの準備

`target_dir`と`source_dir`にそれぞれ変換先の声と変換前の声を入れる。


## 2 モデルの学習

```
poetry run python bin/train.py --source_data_dir "source_dir" --target_data_dir "target_dir" --testdata_dir "test_data" --asr_ckpt_path "models/asr-latest.pt" --vocoder_ckpt_path "models/hifi-gan-latest.pt"
```
