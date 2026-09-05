"""MobileNet backbone used by the A2L adaptation entry."""
from .mobilenet import MobileNetV2


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone != "mobilenet":
        raise ValueError("This A2L project supports backbone='mobilenet' only; got {!r}".format(backbone))
    return MobileNetV2(output_stride, BatchNorm)
