import pyaudio
import numpy as np
import threading
import time
import torch
from src.model.asr_feature_extractor import FeatureExtractor
from src.model.asr_encoder import Encoder
from src.model.bottleneck import Bottleneck
from src.model.hifi_gan_generator import Generator
from src.model.residual_coupling_block import ResidualCouplingBlock

mutex = threading.Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(
    "output/vc/ckpt/exp-20230226-091604/ckpt-latest.pt", map_location=device
)

feature_extractor = FeatureExtractor()
feature_extractor.melspec.set_mode("DFT", "store")
feature_extractor.eval()

encoder = Encoder(240, 128).to(device)
bottleneck = Bottleneck().to(device)
vocoder = Generator().to(device)
flow = ResidualCouplingBlock().to(device)

encoder.load_state_dict(ckpt["asr_encoder"])
bottleneck.load_state_dict(ckpt["bottleneck"])
vocoder.load_state_dict(ckpt["vocoder"])
flow.load_state_dict(ckpt["flow"])

encoder.eval()
bottleneck.eval()
vocoder.eval()
flow.eval()

feature_extractor = FeatureExtractor()
feature_extractor.melspec.set_mode("DFT", "store")
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


asr_history_size = 6 * 256
history_size = 128
vocoder_history_size = 16

# historyを初期化
feat_1_history = torch.zeros((1, asr_history_size, 240)).to(device)
feat_2_history = torch.zeros((1, history_size, 128)).to(device)
feat_3_history = torch.zeros((1, 256, vocoder_history_size)).to(device)

RATE = 24000
CHUNK = 256 * 6
FORMAT = pyaudio.paFloat32
CHANNELS = 1

pa = pyaudio.PyAudio()
print("open")
stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=FORMAT,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)

while True:
    try:
        d = stream.read(CHUNK)

        with torch.no_grad():
            audio = torch.from_numpy(np.frombuffer(d, dtype=np.float32).copy())
            audio = audio.unsqueeze(0).to(device)

            feat = feature_extractor(audio)

            feat_1_history = torch.cat([feat_1_history, feat], dim=1)[
                :, -asr_history_size:, :
            ]

            feat = encoder(feat_1_history)[:, -6:, :]

            feat_2_history = torch.cat([feat_2_history, feat], dim=1)[
                :, -history_size:, :
            ]

            z, _, _ = bottleneck(feat_2_history)
            feat = flow.reverse(z)[:, :, -6:]

            feat_3_history = torch.cat([feat_3_history, feat], dim=2)[
                :, :, -vocoder_history_size:
            ]

            audio_hat = vocoder(feat_3_history)[:, :, -256 * 6 :]
            audio_hat = audio_hat.cpu().detach().numpy()

            stream.write(audio_hat.tobytes())

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        break

stream.stop_stream()
stream.close()
pa.terminate()
