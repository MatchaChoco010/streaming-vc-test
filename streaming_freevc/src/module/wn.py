import torch
from torch import nn
from torch.nn import functional as F


def fused_add_tanh_sigmoid_multiply(input_a, input_b):
    in_act = input_a + input_b
    t_act, s_act = in_act.chunk(2, dim=1)
    t_act = torch.tanh(t_act)
    s_act = torch.sigmoid(s_act)
    acts = t_act * s_act
    return acts


class WN(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation_rate, n_layers, dropout=0.0):
        super(WN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_dim,
                2 * hidden_dim,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_dim
            else:
                res_skip_channels = hidden_dim

            res_skip_layer = nn.Conv1d(hidden_dim, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x):
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts, skip_acts = res_skip_acts.chunk(2, dim=1)
                x = x + res_acts
                output = output + skip_acts
            else:
                output = output + res_skip_acts

        return output

    def remove_weight_norm(self):
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)
