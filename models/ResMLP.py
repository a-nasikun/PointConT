'''
Date: 2022-02-20 07:55:10
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-02-21 02:48:19
'''

import torch
import torch.nn as nn

# ============================================================================
# PAPER <-> CODE MAP: these are the generic MLP building blocks reused
# across Fig. 2 / Fig. 3. See models/PointConT.py and
# models/PointConT_util.py for where each block is instantiated.
# ============================================================================

class MLPBlockFC(nn.Module):
    """Fig. 2 "Cls head": the two fully-connected layers applied after the
    Global MaxPool, before the final `nn.Linear` classifier in
    PointConT_cls (models/PointConT.py)."""
    def __init__(self, d_points, d_model, p_dropout):
        super(MLPBlockFC, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d_points, d_model, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p_dropout))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock2D(nn.Module):
    """Not used in the current PointConT backbone/Inception aggregator path
    (that one uses raw nn.Conv2d + ReLU in PatchAbstraction, Eq. 1);
    provided as a generic Conv2d+BN+LeakyReLU block."""
    def __init__(self, d_points, d_model):
        super(MLPBlock2D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock1D(nn.Module):
    """Fig. 3, final MLP box that fuses f_h (high-freq) and f_l (low-freq)
    branches into f' -- Eq. (4): f' = MLP(||f_h, f_l||). Instantiated as
    `patch_embedding` in Backbone (models/PointConT.py)."""
    def __init__(self, d_points, d_model):
        super(MLPBlock1D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)


class ResMLPBlock1D(nn.Module):
    """Fig. 3, high-frequency aggregation branch -- Eq. (2):
    f_h = ResMLP(MaxPool(f_g)). Instantiated as `self.mlp_res` in
    PatchAbstraction (models/PointConT_util.py) and applied right after the
    MaxPool over each patch's EdgeConv features f_g."""
    def __init__(self, d_points, d_model):
        super(ResMLPBlock1D, self).__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_model),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_points))
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.mlp2(self.mlp1(x)) + x)  # residual MLP: act(MLP(x) + x)
