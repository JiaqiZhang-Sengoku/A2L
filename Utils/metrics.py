"""Dice and ASSD metrics used by the retained two-label evaluation path."""

import medpy.metric.binary as medmetric
import numpy as np
import torch


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    """Compute per-image Dice with the project's historical +1 smoothing."""
    binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=bool)
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(1, 2))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(1, 2))
    intersection = np.sum(intersection.astype(float), axis=(1, 2))
    return (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)


def assd_numpy(binary_segmentation, binary_gt_label):
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)
    if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
        return medmetric.assd(binary_segmentation, binary_gt_label)
    return -1


def dice_coeff_2label(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    return (
        dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]),
        dice_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...]),
    )


def assd_compute(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    assd = np.zeros([pred.shape[0], pred.shape[1]])
    for image_index in range(pred.shape[0]):
        for class_index in range(pred.shape[1]):
            assd[image_index][class_index] = assd_numpy(
                pred[image_index, class_index, ...],
                target[image_index, class_index, ...],
            )
    return assd
