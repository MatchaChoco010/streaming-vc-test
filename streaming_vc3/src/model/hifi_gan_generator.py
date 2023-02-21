import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.causal_conv1d import CausalConv1d
from src.module.res_block import ResBlock
from torch.nn import Upsample
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample_rates = [8, 8, 4]
        self.upsample_kernel_sizes = [16, 16, 8]
        self.upsample_initial_channel = 256
        self.resblock_kernel_sizes = [3, 5, 7]
        self.resblock_dilation_sizes = [[1, 2], [2, 6], [3, 12]]

        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        self.conv_pre = weight_norm(
            CausalConv1d(128, self.upsample_initial_channel, 7, 1)
        )

        self.ups = nn.ModuleList()
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

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(CausalConv1d(ch, 1, 7, 1))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
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
