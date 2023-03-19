import torch
from torch import nn
from torch.nn import functional as F


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.conv_1(self.padding(x))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x))
        return x

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x


class FFT(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers=1,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(
                    hidden_channels,
                    n_heads,
                    dropout=p_dropout,
                    # proximal_bias=proximal_bias,
                    # proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x):
        """
        x: decoder input
        h: encoder output
        """
        for i in range(self.n_layers):
            xt = x.transpose(1, 2)
            y, _ = self.self_attn_layers[i](xt, xt, xt, None)
            y = y.transpose(1, 2)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        return x


class F0Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 1
        self.hidden_channels = 256
        self.filter_channels = 768
        self.n_heads = 2
        self.n_layers = 6
        self.kernel_size = 3
        self.p_dropout = 0.1

        self.prenet = nn.Conv1d(
            self.hidden_channels, self.hidden_channels, 3, padding=1
        )
        self.decoder = FFT(
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        )
        self.proj = nn.Conv1d(self.hidden_channels, self.out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, self.hidden_channels, 3, padding=1)

    def forward(self, x, norm_f0):
        x = torch.detach(x)
        x += self.f0_prenet(norm_f0.unsqueeze(1))[:, :, : x.shape[2]]
        x = self.prenet(x)
        x = self.decoder(x)
        x = self.proj(x)
        return x.squeeze(1)
