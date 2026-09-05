"""Assemble the backbone, multi-scale head and decoder; return logits and features."""

import torch.nn as nn
import torch.nn.functional as F
from .heads.aspp import build_aspp
from .heads.decoder import build_decoder
from .backbone import build_backbone


class SingleDeviceBatchNorm2d(nn.BatchNorm2d):
    """Match the old custom SyncBN's non-parallel path, including its counter state.

    The removed SyncBN implementation delegated single-device computation directly
    to ``F.batch_norm`` and did not increment ``num_batches_tracked``.  Calling the
    same primitive here preserves that behavior while retaining BatchNorm2d's
    identical parameters, buffers, and state-dict keys.
    """

    def forward(self, input):
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


class DeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        # ``sync_bn=True`` is retained for checkpoint/config compatibility. The
        # supported experiments are single-device, where the historical custom
        # SyncBN followed this local BatchNorm path.
        BatchNorm = SingleDeviceBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x, features = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, features

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
