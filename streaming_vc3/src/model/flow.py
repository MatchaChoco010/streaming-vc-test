import torch
from torch import nn
from torch.nn import functional as F

logabs = lambda x: torch.log(torch.abs(x))


# class ActNorm(nn.Module):
#     def __init__(self, in_channel, logdet=True):
#         super().__init__()

#         self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
#         self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

#         self.initialized = False
#         self.logdet = logdet

#     def initialize(self, input):
#         with torch.no_grad():
#             flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
#             mean = (
#                 flatten.mean(1)
#                 .unsqueeze(1)
#                 .unsqueeze(2)
#                 .unsqueeze(3)
#                 .permute(1, 0, 2, 3)
#             )
#             std = (
#                 flatten.std(1)
#                 .unsqueeze(1)
#                 .unsqueeze(2)
#                 .unsqueeze(3)
#                 .permute(1, 0, 2, 3)
#             )

#             self.loc.data.copy_(-mean)
#             self.scale.data.copy_(1 / (std + 1e-6))

#     def forward(self, input):
#         _, _, height, width = input.shape

#         if self.initialized is False:
#             self.initialize(input)
#             self.initialized = True

#         log_abs = logabs(self.scale)

#         logdet = height * width * torch.sum(log_abs)

#         if self.logdet:
#             return self.scale * (input + self.loc), logdet

#         else:
#             return self.scale * (input + self.loc)

#     def reverse(self, output):
#         return output / self.scale - self.loc


# class InvConv1d(nn.Module):
#     def __init__(self, in_channel):
#         super().__init__()

#         weight = torch.randn(in_channel, in_channel)
#         q, _ = torch.qr(weight)
#         # weight = q.unsqueeze(2).unsqueeze(3)
#         weight = q.unsqueeze(2)
#         self.weight = nn.Parameter(weight)

#     def forward(self, input):
#         # _, _, height, width = input.shape
#         _, _, height = input.shape

#         out = F.conv1d(input, self.weight)
#         logdet = (
#             # height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
#             height
#             * torch.slogdet(self.weight.squeeze().double())[1].float()
#         )

#         return out, logdet

#     def reverse(self, output):
#         return F.conv1d(
#             # output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
#             output,
#             self.weight.squeeze().inverse().unsqueeze(2),
#         )


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return torch.sigmoid(out)


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv1d(filter_size, in_channel),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, dim=1)

        log_s, t = self.net(in_a).chunk(2, dim=1)
        # s = torch.exp(log_s)
        s = torch.sigmoid(log_s + 2)
        # out_a = s * in_a + t
        out_b = (in_b + t) * s

        logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        return torch.cat([in_a, out_b], dim=1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, dim=1)

        log_s, t = self.net(out_a).chunk(2, dim=1)
        # s = torch.exp(log_s)
        s = torch.sigmoid(log_s + 2)
        # in_a = (out_a - t) / s
        in_b = out_b / s - t

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # self.actnorm = ActNorm(in_channel)
        # self.invconv = InvConv1d(in_channel)
        self.coupling = AffineCoupling(in_channel)

    def forward(self, input):
        # out, logdet = self.actnorm(input)
        # out, det1 = self.invconv(out)
        # out, det2 = self.coupling(out)
        out, det2 = self.coupling(input)

        # logdet = logdet + det1
        # logdet = logdet + det2
        logdet = det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        # input = self.invconv.reverse(input)
        # input = self.actnorm.reverse(input)

        return input


class FlowModel(nn.Module):
    def __init__(self, n_flows, input_dim):
        super(FlowModel, self).__init__()

        self.n_flows = n_flows
        self.input_dim = input_dim

        self.flow_list = nn.ModuleList()
        for _ in range(n_flows):
            self.flow_list.append(Flow(input_dim))

    def forward(self, x):
        log_det_jacobian = torch.tensor(0.0).to(x.device)
        for flow in self.flow_list:
            x, logdet = flow(x)
            log_det_jacobian += logdet.sum(dim=0)

        return x, log_det_jacobian

    def reverse(self, z):
        for flow in reversed(self.flow_list):
            z = flow.reverse(z)
        return z
