import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.isotonic import IsotonicRegression
import numpy as np


class FcSmall(nn.Module):
    def __init__(self, x_dim=1):
        super(FcSmall, self).__init__()
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100 + 1, 100)
        self.fc3 = nn.Linear(100 + 1, 100)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(100, 2)
        self.iso_transform = None
        self.median = None
        self.group_idx = None
        
    def forward(self, bx, br=None):
        if br is None:
            br = torch.rand(bx.shape[0], 1, device=bx.device)
        h = F.leaky_relu(self.fc1(bx))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc2(h))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc3(h))
        h = self.drop3(h)
        h = self.fc4(h)
        mean = h[:, 0:1]
        stddev = torch.sigmoid(h[:, 1:2]) * 5.0 + 0.01
        return mean, stddev

    def eval_all(self, bx, by):
        eps = 1e-5
        br = torch.rand(bx.shape[0], 1, device=bx.device)
        mean, stddev = self.forward(bx, br)
        cdf = 0.5 * (1.0 + torch.erf((by - mean) / stddev / math.sqrt(2)))

        loss_cdf = torch.abs(cdf - br).mean()
        loss_stddev = stddev.mean()

        # Log likelihood of by under the predicted Gaussian distribution
        loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((by - mean) / stddev) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        return cdf, loss_cdf, loss_stddev, loss_nll

    def recalibrate(self, bx, by, group_idx=-1):
        if group_idx < 0:
            self.iso_transform = self.fit_iso_transform(bx, by) 
        else:
            vals = bx[:, group_idx]
            sorted, indices = vals.sort()
            median = sorted[bx.shape[0] // 2]
            self.median = median
            self.group_idx = group_idx
            self.iso_transform = [
                self.fit_iso_transform(bx[vals <= median], by[vals <= median]),
                self.fit_iso_transform(bx[vals > median], by[vals > median]) 
            ]

    def fit_iso_transform(self, bx, by):
        with torch.no_grad():
            cdf = self.eval_all(bx, by)[0].cpu().numpy()[:, 0].astype(np.float)

        cdf = np.sort(cdf)
        lin = np.linspace(0, 1, int(cdf.shape[0]))

        # Insert an extra 0 and 1 to ensure the range is always [0, 1], and trim CDF for numerical stability
        cdf = np.clip(cdf, a_max=1.0-1e-6, a_min=1e-6)
        cdf = np.insert(np.insert(cdf, -1, 1), 0, 0)
        lin = np.insert(np.insert(lin, -1, 1), 0, 0)

        iso_transform = IsotonicRegression()
        iso_transform.fit_transform(cdf, lin)
        return iso_transform 

    def apply_recalibrate(self, cdf, bx=None):
        if self.iso_transform is not None:
            original_shape = cdf.shape
            if self.median is None:
                return np.reshape(self.iso_transform.transform(cdf.flatten()), original_shape)
            else:
                bx = bx.cpu()
                flattened = cdf.flatten()
                if len(flattened[bx[:, self.group_idx] <= self.median]) > 0:
                    flattened[bx[:, self.group_idx] <= self.median] = self.iso_transform[0].transform(flattened[bx[:, self.group_idx] <= self.median])
                if len(flattened[bx[:, self.group_idx] > self.median]) > 0:
                    flattened[bx[:, self.group_idx] > self.median] = self.iso_transform[1].transform(flattened[bx[:, self.group_idx] > self.median])
                return np.reshape(flattened, original_shape)
        else:
            return cdf

