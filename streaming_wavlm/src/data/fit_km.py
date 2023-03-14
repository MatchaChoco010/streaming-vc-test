import datasets
import joblib
import numpy as np
import os
import torch
import torchaudio
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel


def fit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(  # type: ignore
        "reazon-research/reazonspeech",
        "small",
    )["train"]
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-large").to(device)

    data_len = len(dataset)
    i = 0

    features = []
    for data in dataset:
        i += 1
        print(f"extract feature: {i:4}/{data_len}")

        audio = torch.from_numpy(data["audio"]["array"]).to(
            device=device, dtype=torch.float32
        )
        audio = torchaudio.transforms.Resample(data["audio"]["sampling_rate"], 16000)(
            audio
        )
        audio = audio.unsqueeze(0)

        with torch.no_grad():
            outputs = wavlm(input_values=audio, output_hidden_states=True)

        feat = outputs.hidden_states[10].squeeze(0).cpu().numpy()
        features.append(feat)

    feat = np.concatenate(features, axis=0)

    km_model = MiniBatchKMeans(
        n_clusters=1024,
        batch_size=10000,
        max_no_improvement=100,
        reassignment_ratio=0.0,
        n_init=5,
        verbose=1,
        compute_labels=False,
        init_size=None,
    )

    km_model.fit(feat)

    os.makedirs("output", exist_ok=True)
    joblib.dump(km_model, "output/km_model.km")
