import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.causal_conv1d import CausalConv1d
from src.module.get_padding import get_padding
from src.module.init_weights import init_weights
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock, self).__init__()
        # self.convs = nn.ModuleList(
        #     [
        #         weight_norm(
        #             CausalConv1d(
        #                 channels,
        #                 channels,
        #                 kernel_size,
        #                 1,
        #                 dilation=dilation[0],
        #                 # padding=get_padding(kernel_size, dilation[0]),
        #             )
        #         ),
        #         weight_norm(
        #             CausalConv1d(
        #                 channels,
        #                 channels,
        #                 kernel_size,
        #                 1,
        #                 dilation=dilation[1],
        #                 # padding=get_padding(kernel_size, dilation[1]),
        #             )
        #         ),
        #     ]
        # )
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
