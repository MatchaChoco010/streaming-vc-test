import torch
import torch.nn.functional as F


def feature_loss(fmap_r, fmap_g):
    loss = torch.zeros(1).to(device=fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if len(gl.shape) == 4:
                gl = F.pad(gl, (0, 0, 0, rl.shape[2] - gl.shape[2]), "constant")
            elif len(gl.shape) == 3:
                gl = F.pad(gl, (0, rl.shape[2] - gl.shape[2]), "constant")
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = torch.zeros(1).to(device=disc_real_outputs[0].device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = torch.zeros(1).to(device=disc_outputs[0].device)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
