'''
Date: 2022-03-12 11:47:58
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 14:05:49
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .PointConT_util import PatchAbstraction, ConT
from .ResMLP import MLPBlock1D, MLPBlockFC

# ============================================================================
# PAPER <-> CODE MAP (Point Cloud Classification Using Content-Based
# Transformer via Clustering in Feature Space, Liu et al., IEEE/CAA JAS 2024)
#
# Fig. 2 (Overall architecture): a stack of "Inception feature aggregator"
#   blocks, one per stage, each halving the number of points and doubling
#   the feature dimension, followed by a Global MaxPool + classification
#   head. This is implemented by `Backbone` (the per-stage stack) and
#   `PointConT_cls` (adds the Global MaxPool + Cls head).
#
# Fig. 3 (Inception feature aggregator, per stage m): FPS + KNN -> EdgeConv
#   MLP -> parallel MaxPool/ResMLP (high-freq) and AvgPool/ConT (low-freq)
#   branches -> concat -> MLP. The FPS/KNN/EdgeConv/MaxPool/ResMLP/AvgPool
#   part lives in `PatchAbstraction` (models/PointConT_util.py); the
#   content-based Transformer (ConT) lives in `ConT` (same file); the
#   concat + final MLP (Eq. 4) is the `patch_embedding` step below.
# ============================================================================


class Backbone(nn.Module):
    """
    Implements the stack of "Inception feature aggregator" blocks shown in
    Fig. 2 (Stage 1 ... Stage 5). Each stage i corresponds to one green
    "Inception feature aggregator" box in Fig. 2, built from three
    sub-modules that together realize Fig. 3:
        patch_abstraction[i] -> FPS + KNN + EdgeConv MLP (Eq. 1)
                                 + high-freq branch: MaxPool + ResMLP (Eq. 2)
                                 + low-freq branch input: AvgPool (part of Eq. 3)
        patch_transformer[i] -> ConT, i.e., Softmax((Q-K)/sqrt(d)) (dot) V
                                 applied on the AvgPool output -> f_l (Eq. 3)
        patch_embedding[i]   -> concat(f_h, f_l) + MLP -> f' (Eq. 4)
    """
    def __init__(self, cfg):
        super().__init__()
        self.nblocks = len(cfg.patch_dim) - 1
        self.patch_abstraction = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()
        self.patch_embedding = nn.ModuleList()
        for i in range(self.nblocks):
            # --- Fig. 2 "Stage i" block / Fig. 3 FPS+KNN+EdgeConv+MaxPool+ResMLP+AvgPool ---
            self.patch_abstraction.append(PatchAbstraction(int(cfg.num_points/cfg.down_ratio[i]),
                                                           cfg.patch_size[i],
                                                           2*cfg.patch_dim[i],
                                                           [cfg.patch_dim[i+1], cfg.patch_dim[i+1]]))
            # --- Fig. 3 low-frequency branch: Content-based Transformer (Fig. 4), Eq. (3) ---
            self.patch_transformer.append(ConT(cfg.patch_dim[i+1], cfg.local_size[i], cfg.num_heads))
            # --- Fig. 3 "Linear"/MLP block that fuses f_h and f_l -> f', Eq. (4) ---
            self.patch_embedding.append(MLPBlock1D(cfg.patch_dim[i+1]*2, cfg.patch_dim[i+1]))

    def forward(self, x):
        if x.shape[-1] == 3:
            pos = x
        else:
            pos = x[:, :, :3].contiguous()
        features = x
        pos_and_feats = []
        pos_and_feats.append([pos, features])

        for i in range(self.nblocks):
            # Fig. 3: FPS (downsample centers) + KNN (group patch) + EdgeConv MLP (Eq. 1)
            # -> max_features = high-freq branch output f_h = ResMLP(MaxPool(f_g))  (Eq. 2)
            # -> avg_features = AvgPool(f_g), the *input* to ConT in Eq. (3)
            pos, max_features, avg_features = self.patch_abstraction[i](pos, features)
            # Fig. 3/4: low-freq branch f_l = ConT(AvgPool(f_g))                    (Eq. 3)
            avg_features = self.patch_transformer[i](avg_features)
            # Fig. 3: concatenate high-freq (f_h) and low-freq (f_l) branches, "||f_h, f_l||"
            features = torch.cat([max_features, avg_features], dim=-1)
            # Fig. 3: final MLP of the Inception feature aggregator -> f'          (Eq. 4)
            features = self.patch_embedding[i](features.transpose(1, 2)).transpose(1, 2)
            pos_and_feats.append([pos, features])

        return features, pos_and_feats



class PointConT_cls(nn.Module):
    """
    Full classification network of Fig. 2: the 5-stage `Backbone` (the
    N x 3 -> N/2 x 64 -> ... -> N/32 x 1024 pyramid of Inception feature
    aggregator blocks) followed by the "Global MaxPool" + "Cls head" box
    on the right of Fig. 2.
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)  # Fig. 2: Stage 1 -> Stage 5 (Inception feature aggregators)
        self.mlp1 = MLPBlockFC(cfg.patch_dim[-1], 512, cfg.dropout)   # Fig. 2 "Cls head" (linear layer 1)
        self.mlp2 = MLPBlockFC(512, 256, cfg.dropout)                 # Fig. 2 "Cls head" (linear layer 2)
        self.output_layer = nn.Linear(256, cfg.num_classes)           # final classification layer -> "Airplane" logits

    def forward(self, x):
        patches, _ = self.backbone(x)  # [B, num_patches[-1], patch_dim[-1]] -- Fig. 2 output of Stage 5
        res = torch.max(patches, dim=1)[0]  # Fig. 2 "Global MaxPool" -> [B, patch_dim[-1]]
        res = self.mlp2(self.mlp1(res))     # Fig. 2 "Cls head": two linear layers
        res = self.output_layer(res)        # final class logits (e.g., "Airplane" in Fig. 2)

        return res


