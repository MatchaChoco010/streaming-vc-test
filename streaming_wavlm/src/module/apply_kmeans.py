import joblib
import torch


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)

        center = self.km_model.cluster_centers_.transpose()
        center_norm = (center**2).sum(0, keepdims=True)

        self.center = torch.from_numpy(center).cuda()
        self.center_norm = torch.from_numpy(center_norm).cuda()

    def __call__(self, x):
        dist = (
            x.pow(2).sum(2, keepdim=True)
            - 2 * torch.matmul(x, self.center)
            + self.center_norm
        )
        return dist.argmin(dim=2)
