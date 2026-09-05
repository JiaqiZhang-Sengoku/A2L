"""Positive-only prototype alignment losses for A2L.

Both losses regress pixel features toward a detached class target.  They first
average over pixels within each usable class and then average over usable
classes, so large structures do not dominate small ones.
"""

import torch
import torch.nn.functional as F


_NUM_CLASSES = 3
_NORM_EPS = 1e-12


def _validate_features(features):
    if not torch.is_tensor(features):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 4:
        raise ValueError("features must have shape [B, D, H, W]")


def _validate_targets(targets, name):
    if not isinstance(targets, (list, tuple)) or len(targets) != _NUM_CLASSES:
        raise ValueError("{} must be a list or tuple with three entries".format(name))


def _normalized_detached_target(target, features):
    if target is None:
        return None
    if not torch.is_tensor(target):
        raise TypeError("each class target must be a torch.Tensor or None")

    target = target.detach().to(device=features.device, dtype=features.dtype).reshape(-1)
    if target.numel() != features.size(1):
        raise ValueError(
            "class target dimension {} does not match feature dimension {}".format(
                target.numel(), features.size(1)
            )
        )

    target_norm = target.norm(p=2)
    if (not bool(torch.isfinite(target_norm).item())) or target_norm.item() <= _NORM_EPS:
        return None
    return F.normalize(target, p=2, dim=0, eps=_NORM_EPS)


def _capped_indices(mask_flat, max_pixels_per_class):
    indices = torch.where(mask_flat)[0]
    cap = int(max_pixels_per_class)
    if cap > 0 and indices.numel() > cap:
        permutation = torch.randperm(indices.numel(), device=indices.device)[:cap]
        indices = indices[permutation]
    return indices


def _class_balanced_alignment(features, class_masks, targets, max_pixels_per_class):
    _validate_features(features)
    _validate_targets(targets, "targets")

    expected_mask_shape = (features.size(0), features.size(2), features.size(3))
    feature_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
    class_losses = []
    n_pixels = 0

    for class_id in range(_NUM_CLASSES):
        class_mask = class_masks.get(class_id)
        if class_mask is None:
            continue
        if tuple(class_mask.shape) != expected_mask_shape:
            raise ValueError(
                "class mask {} must have shape {}, got {}".format(
                    class_id, expected_mask_shape, tuple(class_mask.shape)
                )
            )

        target = _normalized_detached_target(targets[class_id], features)
        if target is None:
            continue

        mask_flat = class_mask.to(device=features.device, dtype=torch.bool).reshape(-1)
        indices = _capped_indices(mask_flat, max_pixels_per_class)
        if indices.numel() == 0:
            continue

        selected = feature_flat.index_select(0, indices)
        selected = F.normalize(selected, p=2, dim=1, eps=_NORM_EPS)
        squared_distance = 0.5 * (selected - target.unsqueeze(0)).pow(2).sum(dim=1)
        class_losses.append(squared_distance.mean())
        n_pixels += int(indices.numel())

    if not class_losses:
        return features.sum() * 0.0, 0
    return torch.stack(class_losses).mean(), n_pixels


def prototype_alignment_loss(
    features,
    pseudo_cls,
    prototypes,
    valid_mask=None,
    max_pixels_per_class=2048,
):
    """Align pseudo-labeled pixels with their detached pseudo/reference target.

    Args:
        features: Pixel features with shape ``[B, D, H, W]``.
        pseudo_cls: Integer pseudo labels with shape ``[B, H, W]``.
        prototypes: Three class targets, each ``[D]`` or ``None``.
        valid_mask: Optional boolean mask with shape ``[B, H, W]``.
        max_pixels_per_class: Positive values cap sampled pixels per class;
            non-positive values keep every valid pixel.

    Returns:
        ``(loss, n_pixels)`` where ``n_pixels`` is counted after capping.
    """
    _validate_features(features)
    _validate_targets(prototypes, "prototypes")
    expected_shape = (features.size(0), features.size(2), features.size(3))
    if tuple(pseudo_cls.shape) != expected_shape:
        raise ValueError("pseudo_cls must have shape {}, got {}".format(expected_shape, tuple(pseudo_cls.shape)))

    pseudo_cls = pseudo_cls.to(device=features.device)
    if valid_mask is None:
        valid_mask = torch.ones(expected_shape, dtype=torch.bool, device=features.device)
    else:
        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError(
                "valid_mask must have shape {}, got {}".format(expected_shape, tuple(valid_mask.shape))
            )
        valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)

    class_masks = {
        class_id: (pseudo_cls == class_id) & valid_mask
        for class_id in range(_NUM_CLASSES)
    }
    return _class_balanced_alignment(
        features,
        class_masks,
        prototypes,
        max_pixels_per_class=max_pixels_per_class,
    )


def anchor_regression_loss(
    features,
    class_masks,
    golden_anchors,
    max_pixels_per_class=2048,
):
    """Regress ground-truth-class pixels toward detached golden anchors.

    Args:
        features: Pixel features with shape ``[B, D, H, W]``.
        class_masks: Mapping from class IDs 0, 1, and 2 to ``[B, H, W]`` masks.
        golden_anchors: Three class anchors, each ``[D]`` or ``None``.
        max_pixels_per_class: Positive values cap sampled pixels per class;
            non-positive values keep every valid pixel.

    Returns:
        ``(loss, n_pixels)`` where ``n_pixels`` is counted after capping.
    """
    if not isinstance(class_masks, dict):
        raise TypeError("class_masks must be a dict keyed by class ID")
    return _class_balanced_alignment(
        features,
        class_masks,
        golden_anchors,
        max_pixels_per_class=max_pixels_per_class,
    )
