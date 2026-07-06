'''
Date: 2022-03-11 11:01:07
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 10:01:25
'''

import math
from einops import rearrange
import torch
import torch.nn as nn

from pointnet2_ops import pointnet2_utils

from pointnet_util import index_points, square_distance
from .ResMLP import ResMLPBlock1D


def Point2Patch(num_patches, patch_size, xyz):
    """
    Patch Partition in 3D Space.

    Fig. 3 mapping: this is the "FPS" box (downsamples center points i at
    2x rate, i.e., num_patches = N/2^m) followed by the "KNN" box (groups
    the surrounding patch {j : (i,j) in N(i)} of size k = patch_size around
    each center point i). Search is done in 3D coordinate space, as stated
    in the text ("we adopt neighbor search in the 3D space", unlike DGCNN
    which searches in feature space).

    Input:
        num_patches: number of patches, S (= N/2^m in the paper's notation)
        patch_size: number of points per patch, k
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        knn_idx: [B, S, k]
    """
    # Fig. 3 "FPS": downsample center points i via furthest point sampling
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_patches).long()  # [B, S]
    centroid_xyz = index_points(xyz, fps_idx)    # [B, S, 3]
    # Fig. 3 "KNN": group the k=patch_size neighbors j in N(i) around each center i
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k]

    return centroid_xyz, fps_idx, knn_idx



class PatchAbstraction(nn.Module):
    """
    Implements the *left/center* part of Fig. 3 (everything up to, but not
    including, the content-based attention box): FPS + KNN -> EdgeConv-style
    MLP (Eq. 1) -> the parallel high-frequency branch (MaxPool + ResMLP,
    Eq. 2) and the average-pooling half of the low-frequency branch
    (AvgPool, the input to ConT in Eq. 3). The ConT itself is applied
    outside this module (see `Backbone.forward` in PointConT.py).
    """
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PatchAbstraction, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()
        # Fig. 3 high-freq branch "Conv1d/BatchNorm/LeakyReLU" residual box -> ResMLP in Eq. (2)
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel
        for out_channel in mlp:
            # Fig. 3 "Conv2d / BatchNorm / ReLU x2" box: the EdgeConv-style MLP of Eq. (1)
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz, feature):
        """
        Input: xyz [B, S_, 3]
               features [B, S_, C]
        Return: [B, S, 3+D]
        """
        B, _, C = feature.shape
        # Fig. 3 "FPS" + "KNN": get center indices (centroid_idx) and their
        # patch neighbors {j : (i,j) in N(i)} (knn_idx)
        centroid_xyz, centroid_idx, knn_idx = Point2Patch(self.num_patches, self.patch_size, xyz)

        centroid_feature = index_points(feature, centroid_idx) # f_i, [B, S, C]  (center/query point features)
        grouped_feature = index_points(feature, knn_idx)    # f_j, [B, S, k, C]  (neighbor features, j in N(i))

        k = grouped_feature.shape[2]

        # Eq. (1): f_j - f_i, the neighbor features relative to the centroid
        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C) # [B, S, k, C]
        # Eq. (1): || f_i , f_j - f_i || concatenation (the "||" operator in the paper)
        groups = torch.cat((centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C), grouped_norm), dim=-1) # [B, S, k, 2C]

        groups = groups.permute(0, 3, 2, 1) # [B, Channel, k, S]

        # Eq. (1): f_g = MLP(|| f_i, f_j - f_i ||)  -- Fig. 3 "Conv2d/BatchNorm/ReLU x2" box
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            groups =  act(bn(conv(groups))) # [B, D, k, S] == f_g

        # Eq. (2), high-frequency aggregation branch: f_h = ResMLP(MaxPool(f_g))
        max_patches = torch.max(groups, 2)[0] # Fig. 3 "MaxPool" -> [B, D, S]
        max_patches = self.mlp_res(max_patches).transpose(1, 2) # Fig. 3 "ResMLP" -> f_h, [B, S, D]

        # Eq. (3), low-frequency aggregation branch (part 1 of 2):
        # f_l = ConT(AvgPool(f_g)) -- AvgPool computed here, ConT applied by the caller
        avg_patches = torch.mean(groups, 2).transpose(1, 2) # Fig. 3 "AvgPool" -> [B, S, D]

        return centroid_xyz, max_patches, avg_patches



class ConT(nn.Module):
    '''
    Content-based Transformer -- implements Fig. 4 and Section III-C of the
    paper ("Content-Based Transformer"). local_size == |Qi| == S/L is the
    size of each cluster (the paper's "local cluster size", ablated in
    Table V), and the number of clusters L = S / local_size, with
    n = log2(L) hierarchical binary-clustering iterations (Eq. 6).

    Args:
        dim (int): Number of input channels.
        local_size (int): The size of the local feature space.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, local_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., kmeans = False):

        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        '''
        Input: [B, S, D]
        Return: [B, S, D]

        Implements Fig. 4 ("Illustration of content-based attention") and
        Section III-C. Overall flow: X -> Q,K,V embeddings -> hierarchical
        binary clustering of the queries (Sec. III-C.1, Eq. 6) -> "share
        index" gather of K,V by the same cluster assignment -> per-cluster
        vector attention (Eq. 8) -> merge clusters back to the original
        point order -> output projection + residual connection.
        '''

        B, S, D = x.shape
        nl = S // self.ls  # L = S / |Qi| = number of clusters (Sec. III-C: "L subsets {Qi}, |Qi| = S/L")
        # Fig. 4 "Q, K, V": Q = X W_Q, K = X W_K, V = X W_V (linear embeddings, just above Eq. 5)
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, h, S, d]

        q_pre = qkv[0].reshape(B*self.num_heads, S, D // self.num_heads).permute(0,2,1) # [B*h, d, S]
        ntimes = int(math.log(nl, 2))  # n = log2(L), Sec. III-C.1 ("After performing n iterations (n = log2 L)")
        q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B*self.num_heads, S)

        # ------------------------------------------------------------------
        # Fig. 4 "Cluster" box / Sec. III-C.1 "Hierarchical Binary Clustering", Eq. (6):
        #   c1 = mean of first half, c2 = mean of second half   (cluster centroids)
        #   r_i = dist(q_i, c1) / dist(q_i, c2)                 (distance ratio)
        #   [i1,...,iS] = argsort({r_i})                        (hard assignment via sort)
        #   C1 = first half of sorted queries, C2 = second half (the two balanced clusters)
        # This is repeated n = log2(L) times, recursively splitting each
        # cluster into two, until L equal-size clusters are obtained.
        # ------------------------------------------------------------------
        for _ in range(ntimes):
            bh,d,n = q_pre.shape # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1) # [B*h*2^n, d, 2]  -- Eq. (6): c1, c2 (centroids of the 2 halves)

            q_avg = torch.nn.functional.normalize(q_avg.permute(0,2,1), dim=-1)
            q_norm = torch.nn.functional.normalize(q_pre.permute(0,2,1), dim=-1)

            q_scores = square_distance(q_norm, q_avg) # [B*h*2^n, S/2^n, 2] -- dist(qi,c1), dist(qi,c2), Eq. (6)
            q_ratio = (q_scores[:,:,0]+1) / (q_scores[:,:,1]+1) # Eq. (6): r_i = dist(qi,c1)/dist(qi,c2)
            q_idx = q_ratio.argsort()  # Eq. (6): [i1,...,iS] = argsort({r_i}) -- the hard cluster assignment

            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh*2, n//2) # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size()) # [B*h*2^n, d, S/2^n]
            # Eq. (6): C1 = first S/2 sorted queries, C2 = last S/2 sorted queries (split in half)
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')   # [B*h*2^(n+1), d, S/(2^(n+1))] -- recurse into each new cluster

        # "Multi-head configuration... each head performs query/key/value
        # embeddings and clustering independently" (Sec. III-C, Fig. 5)
        q_idx = q_idx_last.view(B,self.num_heads, S) # [B, h, S] -- final permutation grouping points by cluster
        q_idx_rev = q_idx.argsort() # [B, h, S] -- inverse permutation, used later to restore original order

        # Fig. 4 "Share index": K and V are gathered using the *same* cluster
        # assignment computed from Q, i.e. {Ki}, {Vi} are split by the same
        # index as {Qi} (text just above Eq. 5)
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size()) # [3, B, h, S, d]
        qkv_pre = qkv.gather(dim=-2, index=q_idx) # [3, B, h, S, d]
        # split the clustered sequence into L = nl non-overlapping clusters of size ls = |Qi|
        q, k, v  = rearrange(qkv_pre, 'qkv b h (nl ls) d -> qkv (b nl) h ls d', ls=self.ls)

        # Eq. (8), vector attention (the SA choice adopted in this paper,
        # Sec. III-C.2): SA = Softmax((Q-K)/sqrt(d)) (dot) V, computed
        # independently within each cluster/subset Qi -> Eq. (5): Yi = SA(Qi,Ki,Vi)
        attn = (q - k)*self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out =  torch.einsum('bhld, bhld->bhld', attn, v) # [B*(nl), h, ls, d] -- elementwise (dot) V of Eq. (8)

        # Fig. 4 "merge and reverse" / Sec. III-C: "all subsets {Yi} are
        # merged into the output Y in keeping with their original order"
        out = rearrange(out, '(b nl) h ls d -> b h d (nl ls)', h=self.num_heads, b=B) # [B, h, d, S] -- merge clusters
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1,index=q_idx_rev).reshape(B,D,S).permute(0,2,1) # [B, S, D] -- reverse to original point order

        res = self.proj(res) # output projection (Fig. 3 "Linear" box feeding into the residual add)
        res = self.proj_drop(res)

        res = x + res # residual connection around the content-based attention (Fig. 3 "+" node)

        return res


