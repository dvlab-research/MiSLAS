
"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.
Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):  # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


def create_loss (cls_num_list=None, reweight_CE=False):
    print('Loading Softmax Loss.')
    return CrossEntropyLoss(cls_num_list, reweight_CE).cuda()