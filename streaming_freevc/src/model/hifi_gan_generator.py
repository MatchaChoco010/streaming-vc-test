import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.causal_conv1d import CausalConv1d
from src.module.res_block import ResBlock
from torch.nn import Upsample
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


def pad_diff(x):
    return F.pad(
        F.pad(x, (0, 0, -1, 1), "constant", 0) - x, (0, 0, 0, -1), "constant", 0
    )


class SineGen(nn.Module):
    def __init__(self):
        super(SineGen, self).__init__()
        self.sine_amp = 0.1
        self.noise_std = 0.003
        self.harmonic_num = 8
        self.dim = self.harmonic_num + 1
        self.sampling_rate = 16000
        self.voiced_threshold = 0

        self.linear = nn.Linear(self.harmonic_num + 1, 1)
        self.tanh = nn.Tanh()

    def _f0_to_uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f0_to_sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_init = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_init[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_init

        tmp_over_one = torch.cumsum(rad_values, dim=1) % 1
        tmp_over_one_idx = (pad_diff(tmp_over_one)) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0):
        with torch.no_grad():
            fn = torch.multiply(
                f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
            )

            sine_waves = self._f0_to_sine(fn) * self.sine_amp

            uv = self._f0_to_uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

            sine_merge = self.tanh(self.linear(sine_waves))

            return sine_merge


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.upsample_rates = [8, 8, 4]
        # self.upsample_kernel_sizes = [16, 16, 8]
        # self.upsample_initial_channel = 256
        # self.resblock_kernel_sizes = [3, 5, 7]
        # self.resblock_dilation_sizes = [[1, 2], [2, 6], [3, 12]]
        self.upsample_rates = [10, 8, 2, 2]
        self.upsample_initial_channel = 512
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        self.source = SineGen()
        self.f0_upsample = torch.nn.Upsample(scale_factor=320)

        self.conv_pre = weight_norm(
            CausalConv1d(256, self.upsample_initial_channel, 7, 1)
        )

        self.ups = nn.ModuleList()
        self.source_conv = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.Sequential(
                    Upsample(scale_factor=u),
                    weight_norm(
                        CausalConv1d(
                            self.upsample_initial_channel // (2**i),
                            self.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                        )
                    ),
                )
            )
            if i != len(self.upsample_rates) - 1:
                stride_f0 = self.upsample_rates[i + 1]
                self.source_conv.append(
                    CausalConv1d(
                        int(self.upsample_initial_channel // (2 ** (i + 2))),
                        int(self.upsample_initial_channel // (2 ** (i + 1))),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        # padding=stride_f0 // 2,
                    )
                )
            else:
                self.source_conv.append(
                    CausalConv1d(
                        1,
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=1,
                    )
                )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(CausalConv1d(ch, 1, 7, 1))

    def forward(self, x, f0):
        f0 = self.f0_upsample(f0[:, None]).transpose(1, 2)
        source_wave = self.source(f0).transpose(1, 2)
        source_waves = []
        for i in range(len(self.source_conv)):
            source_wave = self.source_conv[-(i + 1)](source_wave)
            source_waves.append(source_wave)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = source_waves[-(i + 1)]
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
