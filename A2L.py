"""A2L adaptation: UPA prototypes, PAUH querying, and LGMC calibration."""
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable

from Utils.project_paths import resolve_path
from Utils.runtime import (parse_args, effective_config, create_run, write_json, write_csv,
                           checkpoint_payload, load_checkpoint, sha256_file, apply_validation_split,
                           experiment_domains)


def load_runtime_dependencies():
    """Keep --help and --print-config usable before installing ML dependencies."""
    global np, torch, cudnn, F, DataLoader, Subset, transforms
    global trans, fundus_dataloader2, netd, assd_compute, dice_coeff_2label
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from Dataloaders import transforms as trans
    from Dataloaders import fundus as fundus_dataloader2
    import Networks.deeplabv3 as netd
    from Utils.metrics import assd_compute, dice_coeff_2label


def setup_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_name(name_or_path):
    return os.path.basename(str(name_or_path).replace("\\", "/"))


def compute_active_num(total_count, in_args):
    return min(total_count, in_args.active_num)


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    return (soft_pls > pseudo_label_threshold).float()


def dice_loss_multichannel(probs, target, eps=1e-6):
    intersection = torch.sum(probs * target, dim=(0, 2, 3))
    cardinality = torch.sum(probs + target, dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - torch.mean(dice)


def compute_class_balance_weight(pred_bank, in_args):
    if len(pred_bank) == 0:
        return 1.0

    device = next(iter(pred_bank.values())).device
    not_cup_loss_sum = torch.zeros(1, device=device)
    cup_loss_sum = torch.zeros(1, device=device)
    not_cup_loss_num = torch.zeros(1, device=device)
    cup_loss_num = torch.zeros(1, device=device)

    lower_bound = in_args.pseudo_label_threshold * 0.2
    upper_bound = 1.0 - ((1.0 - in_args.pseudo_label_threshold) * 0.2)
    eps = 1e-6

    for pred_i in pred_bank.values():
        cup_pred = pred_i[0, ...].clamp(min=eps, max=1.0 - eps)
        neg_mask = (cup_pred < in_args.pseudo_label_threshold) & (cup_pred > lower_bound)
        pos_mask = (cup_pred > in_args.pseudo_label_threshold) & (cup_pred < upper_bound)

        not_cup_loss_sum += torch.sum(-torch.log(1.0 - cup_pred[neg_mask]))
        cup_loss_sum += torch.sum(-torch.log(cup_pred[pos_mask]))
        not_cup_loss_num += torch.sum(neg_mask)
        cup_loss_num += torch.sum(pos_mask)

    if not_cup_loss_num.item() <= 0 or cup_loss_num.item() <= 0:
        return 1.0

    mean_not_cup = not_cup_loss_sum / (not_cup_loss_num + eps)
    mean_cup = cup_loss_sum / (cup_loss_num + eps)
    weight = (mean_cup / (mean_not_cup + eps)).item()

    if (not np.isfinite(weight)) or (weight <= 0):
        return 1.0
    return float(weight)


def to_exclusive_three_class_probs(prob_2ch, eps=1e-6):
    cup = prob_2ch[:, 0:1, ...]
    disc = prob_2ch[:, 1:2, ...]

    p_cup = torch.clamp(cup, min=0.0, max=1.0)
    p_disc_only = torch.clamp(disc - cup, min=0.0, max=1.0)
    p_bg = torch.clamp(1.0 - disc, min=0.0, max=1.0)

    probs = torch.cat([p_bg, p_cup, p_disc_only], dim=1)
    probs = probs / torch.sum(probs, dim=1, keepdim=True).clamp_min(eps)
    return probs


def pseudo_class_from_prob2ch(prob_2ch, target_size=None):
    probs3 = to_exclusive_three_class_probs(prob_2ch)
    if target_size is not None and probs3.shape[2:] != target_size:
        probs3 = F.interpolate(probs3, size=target_size, mode="bilinear", align_corners=True)
        probs3 = probs3 / torch.sum(probs3, dim=1, keepdim=True).clamp_min(1e-6)
    confidence, pseudo_cls = torch.max(probs3, dim=1)
    return pseudo_cls, confidence, probs3


def build_exclusive_masks_from_gt(mask_2ch):
    cup = mask_2ch[:, 0, ...] > 0.5
    disc = mask_2ch[:, 1, ...] > 0.5
    disc_only = disc & (~cup)
    bg = (~disc) & (~cup)
    return {0: bg, 1: cup, 2: disc_only}


def build_exclusive_masks_from_probs(prob_2ch, conf_thr=0.9, target_size=None):
    pseudo_cls, confidence, _ = pseudo_class_from_prob2ch(prob_2ch, target_size=target_size)
    return {
        0: (pseudo_cls == 0) & (confidence >= conf_thr),
        1: (pseudo_cls == 1) & (confidence >= conf_thr),
        2: (pseudo_cls == 2) & (confidence >= conf_thr),
    }


def update_prototypes_ema(prototypes, features, class_masks, momentum):
    counts = [0, 0, 0]
    for cls_id in (0, 1, 2):
        mask = class_masks[cls_id].unsqueeze(1).float()
        count = torch.sum(mask)
        counts[cls_id] = int(count.item())
        if count.item() <= 0:
            continue

        proto_batch = torch.sum(features * mask, dim=(0, 2, 3)) / (count + 1e-6)
        proto_batch = proto_batch.detach()
        if prototypes[cls_id] is None:
            prototypes[cls_id] = proto_batch
        else:
            prototypes[cls_id] = momentum * prototypes[cls_id] + (1.0 - momentum) * proto_batch
    return counts


def get_reference_prototypes(golden_anchors, pseudo_prototypes):
    refs = [None, None, None]
    for cls_id in (0, 1, 2):
        if golden_anchors[cls_id] is not None:
            refs[cls_id] = golden_anchors[cls_id]
        else:
            refs[cls_id] = pseudo_prototypes[cls_id]
    return refs


def prototype_classification_loss(
    features,
    pseudo_cls,
    prototypes,
    temperature,
    valid_mask=None,
    max_pixels_per_class=2048,
):
    available = [cls_id for cls_id in (0, 1, 2) if prototypes[cls_id] is not None]
    if len(available) < 2:
        return features.new_tensor(0.0), 0

    feat = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
    cls_flat = pseudo_cls.contiguous().view(-1)
    if valid_mask is not None:
        valid_flat = valid_mask.contiguous().view(-1)
    else:
        valid_flat = torch.ones_like(cls_flat, dtype=torch.bool)

    class_to_local = {cls_id: i for i, cls_id in enumerate(available)}
    query_list = []
    target_list = []

    for cls_id in (0, 1, 2):
        if cls_id not in class_to_local:
            continue
        idx = torch.where((cls_flat == cls_id) & valid_flat)[0]
        if idx.numel() == 0:
            continue
        if (max_pixels_per_class > 0) and (idx.numel() > max_pixels_per_class):
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_pixels_per_class]
            idx = idx[perm]
        query_list.append(feat[idx])
        target_list.append(torch.full((idx.numel(),), class_to_local[cls_id], dtype=torch.long, device=feat.device))

    if len(query_list) == 0:
        return features.new_tensor(0.0), 0

    queries = torch.cat(query_list, dim=0)
    targets = torch.cat(target_list, dim=0)

    proto_mat = torch.stack([prototypes[cls_id] for cls_id in available], dim=0)
    queries = F.normalize(queries, dim=1, p=2, eps=1e-12)
    proto_mat = F.normalize(proto_mat, dim=1, p=2, eps=1e-12)
    logits = torch.matmul(queries, proto_mat.t()) / max(temperature, 1e-6)
    loss = F.cross_entropy(logits, targets)
    return loss, int(queries.size(0))


def compute_class_adaptive_margins(class_hist, margin_scale, eps=1e-6):
    hist = class_hist.float().clamp_min(eps)
    freq = hist / torch.sum(hist).clamp_min(eps)
    inv = torch.rsqrt(freq + eps)
    inv = inv / torch.mean(inv).clamp_min(eps)
    return margin_scale * inv


def calibration_contrastive_loss(
    features,
    class_masks,
    golden_anchors,
    margins,
    temperature,
    max_pixels_per_class=2048,
):
    available = [cls_id for cls_id in (0, 1, 2) if golden_anchors[cls_id] is not None]
    if len(available) < 2:
        return features.new_tensor(0.0), 0

    feat = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
    class_to_local = {cls_id: i for i, cls_id in enumerate(available)}
    query_list = []
    target_list = []

    for cls_id in (0, 1, 2):
        if cls_id not in class_to_local:
            continue
        mask = class_masks[cls_id].contiguous().view(-1)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            continue
        if (max_pixels_per_class > 0) and (idx.numel() > max_pixels_per_class):
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_pixels_per_class]
            idx = idx[perm]
        query_list.append(feat[idx])
        target_list.append(torch.full((idx.numel(),), class_to_local[cls_id], dtype=torch.long, device=feat.device))

    if len(query_list) == 0:
        return features.new_tensor(0.0), 0

    queries = torch.cat(query_list, dim=0)
    targets = torch.cat(target_list, dim=0)

    anchor_mat = torch.stack([golden_anchors[cls_id] for cls_id in available], dim=0)
    queries = F.normalize(queries, dim=1, p=2, eps=1e-12)
    anchor_mat = F.normalize(anchor_mat, dim=1, p=2, eps=1e-12)

    tau = max(temperature, 1e-6)
    logits = torch.matmul(queries, anchor_mat.t()) / tau
    local_margin = torch.tensor([float(margins[cls_id]) for cls_id in available], dtype=logits.dtype, device=logits.device)
    row_idx = torch.arange(logits.size(0), device=logits.device)
    logits[row_idx, targets] = logits[row_idx, targets] - local_margin[targets] / tau
    loss = F.cross_entropy(logits, targets)
    return loss, int(queries.size(0))


@dataclass(frozen=True)
class Policy:
    """Fixed internal implementation choices; not CLI or research parameters."""
    method_variant: str
    warmup_sca: Callable
    guided_sca: Callable
    guided_cal: Callable
    golden_update_timing: str
    inactive_parameters: tuple = ()
    ema_rule: str = "fixed"


def _classification_sca(features, pseudo_cls, prototypes, in_args, *, valid_mask, max_pixels_per_class):
    return prototype_classification_loss(
        features, pseudo_cls, prototypes, in_args.temperature,
        valid_mask=valid_mask, max_pixels_per_class=max_pixels_per_class)


def _classification_cal(features, class_masks, golden_anchors, margins, in_args, *, max_pixels_per_class):
    return calibration_contrastive_loss(
        features, class_masks, golden_anchors, margins, in_args.temperature,
        max_pixels_per_class=max_pixels_per_class)


def _regression_warmup_sca(features, pseudo_cls, prototypes, in_args, *, valid_mask, max_pixels_per_class):
    # Preserve the historical warm-up zero scalar and skip its sampling at weight zero.
    loss, n_pixels = features.sum() * 0.0, 0
    if in_args.beta_sca != 0:
        from Utils.prototype_alignment import prototype_alignment_loss
        loss, n_pixels = prototype_alignment_loss(
            features, pseudo_cls, prototypes, valid_mask=valid_mask,
            max_pixels_per_class=max_pixels_per_class)
    return loss, n_pixels


def _regression_guided_sca(features, pseudo_cls, prototypes, in_args, *, valid_mask, max_pixels_per_class):
    if in_args.beta_sca == 0:
        return torch.zeros(1, device=features.device), 0
    from Utils.prototype_alignment import prototype_alignment_loss
    return prototype_alignment_loss(
        features, pseudo_cls, prototypes, valid_mask=valid_mask,
        max_pixels_per_class=max_pixels_per_class)


def _regression_cal(features, class_masks, golden_anchors, margins, in_args, *, max_pixels_per_class):
    if in_args.lambda_cal == 0:
        return torch.zeros(1, device=features.device), 0
    from Utils.prototype_alignment import anchor_regression_loss
    return anchor_regression_loss(
        features, class_masks, golden_anchors, max_pixels_per_class=max_pixels_per_class)


CLASSIFICATION = Policy(
    "prototype_classification", _classification_sca, _classification_sca,
    _classification_cal, "before_loss")
REGRESSION_LAGGED = Policy(
    "a2l_positive_prototype_regression", _regression_warmup_sca, _regression_guided_sca,
    _regression_cal, "after_step", ("temperature", "cal_margin_scale"))


def _resolve_policy(policy):
    if policy is None:
        return CLASSIFICATION
    if not any(policy is fixed for fixed in (CLASSIFICATION, REGRESSION_LAGGED)):
        raise ValueError("choose one of the fixed internal A2L policies")
    return policy


def _ema_decay_stats(decays):
    return {
        "ema_decay_mean": sum(decays) / len(decays) if decays else None,
        "ema_decay_min": min(decays) if decays else None,
        "ema_decay_max": max(decays) if decays else None,
    }


def _fetch_next(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def init_pred_bank(model, loader, device):
    pred_bank = {}
    model_mode = model.training
    model.eval()
    with torch.no_grad():
        for sample in loader:
            data = sample["image"].to(device, non_blocking=True)
            img_names = [normalize_name(name) for name in sample["img_name"]]

            pred, _ = model(data)
            pred = torch.sigmoid(pred)
            for i in range(data.size(0)):
                pred_bank[img_names[i]] = pred[i].detach().clone()

    if model_mode:
        model.train()
    return pred_bank


def bootstrap_golden_anchors(model_s, labeled_loader, golden_anchors, device):
    if len(labeled_loader) <= 0:
        return [0, 0, 0]

    model_mode = model_s.training
    model_s.eval()

    feat_sums = [None, None, None]
    feat_counts = [0.0, 0.0, 0.0]

    with torch.no_grad():
        for _, sample_s in labeled_loader:
            imgs_s = sample_s["image"].to(device, non_blocking=True)
            labels_s = sample_s["label"].to(device, non_blocking=True)

            _, feat_s = model_s(imgs_s)
            labels_low = F.interpolate(labels_s, size=feat_s.shape[2:], mode="nearest")
            masks = build_exclusive_masks_from_gt(labels_low)

            for cls_id in (0, 1, 2):
                mask = masks[cls_id].unsqueeze(1).float()
                count = float(torch.sum(mask).item())
                if count <= 0:
                    continue
                feat_sum = torch.sum(feat_s * mask, dim=(0, 2, 3)).detach()
                if feat_sums[cls_id] is None:
                    feat_sums[cls_id] = feat_sum
                else:
                    feat_sums[cls_id] = feat_sums[cls_id] + feat_sum
                feat_counts[cls_id] += count

    for cls_id in (0, 1, 2):
        if feat_counts[cls_id] > 0:
            golden_anchors[cls_id] = (feat_sums[cls_id] / feat_counts[cls_id]).detach()

    if model_mode:
        model_s.train()
    return [int(v) for v in feat_counts]


def estimate_labeled_class_hist(labeled_loader, device):
    hist = torch.ones(3, dtype=torch.float32, device=device)
    if len(labeled_loader) <= 0:
        return hist

    for _, sample_s in labeled_loader:
        labels = sample_s["label"].to(device, non_blocking=True)
        masks = build_exclusive_masks_from_gt(labels)
        hist[0] += torch.sum(masks[0]).float()
        hist[1] += torch.sum(masks[1]).float()
        hist[2] += torch.sum(masks[2]).float()
    return hist


def adapt_epoch_warmup(
    model_t,
    model_s,
    optim,
    train_loader,
    in_args,
    pred_bank,
    pseudo_prototypes,
    *,
    policy=None,
):
    policy = _resolve_policy(policy)
    if len(train_loader) <= 0:
        return {
            "loss_total": 0.0,
            "loss_seg": 0.0,
            "loss_sca": 0.0,
            "loss_weight": 1.0,
            "proto_pixels": [0, 0, 0],
            "sca_pixels": 0,
            "num_steps": 0,
            **_ema_decay_stats(()),
        }

    device = next(model_s.parameters()).device
    loss_weight = compute_class_balance_weight(pred_bank, in_args)

    loss_total_sum = 0.0
    loss_seg_sum = 0.0
    loss_sca_sum = 0.0
    proto_pixels = [0, 0, 0]
    sca_pixels = 0
    num_steps = 0
    ema_decays = []

    for sample_w, sample_s in train_loader:
        imgs_w = sample_w["image"].to(device, non_blocking=True)
        imgs_s = sample_s["image"].to(device, non_blocking=True)
        names = [normalize_name(name) for name in sample_w["img_name"]]

        optim.zero_grad(set_to_none=True)

        logits_s, feat_s = model_s(imgs_s)
        probs_s = torch.sigmoid(logits_s)
        with torch.no_grad():
            logits_t, feat_t = model_t(imgs_w)
            probs_t = torch.sigmoid(logits_t)

        pseudo_2ch = soft_label_to_hard(probs_t, in_args.pseudo_label_threshold)
        loss_seg_pixel = F.binary_cross_entropy_with_logits(logits_s, pseudo_2ch, reduction="none")
        loss_mask = torch.ones_like(pseudo_2ch)
        loss_mask[:, 0, ...][pseudo_2ch[:, 0, ...] == 0] = loss_weight
        loss_seg_bce = torch.sum(loss_seg_pixel * loss_mask) / torch.sum(loss_mask).clamp_min(1e-6)
        loss_seg_dice = dice_loss_multichannel(probs_s, pseudo_2ch)
        loss_seg = loss_seg_bce + loss_seg_dice

        masks_t = build_exclusive_masks_from_probs(
            probs_t, conf_thr=0.995, target_size=feat_t.shape[2:]
        )
        counts = update_prototypes_ema(pseudo_prototypes, feat_t.detach(), masks_t, 0.95)
        for cls_id in (0, 1, 2):
            proto_pixels[cls_id] += counts[cls_id]

        pseudo_cls_s, conf_s, _ = pseudo_class_from_prob2ch(probs_t, target_size=feat_s.shape[2:])
        valid_mask_s = conf_s >= 0.95
        loss_sca, num_sca = policy.warmup_sca(
            feat_s,
            pseudo_cls_s,
            pseudo_prototypes,
            in_args,
            valid_mask=valid_mask_s,
            max_pixels_per_class=512,
        )
        sca_pixels += num_sca

        loss = loss_seg + in_args.beta_sca * loss_sca
        loss.backward()
        optim.step()

        with torch.no_grad():
            for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
                param_t.data = param_t.data * in_args.model_ema_rate + param_s.data * (1.0 - in_args.model_ema_rate)
        ema_decays.append(float(in_args.model_ema_rate))

        for idx, name in enumerate(names):
            pred_bank[name] = probs_t[idx].detach().clone()

        loss_total_sum += float(loss.item())
        loss_seg_sum += float(loss_seg.item())
        loss_sca_sum += float(loss_sca.item())
        num_steps += 1

    return {
        "loss_total": loss_total_sum / max(num_steps, 1),
        "loss_seg": loss_seg_sum / max(num_steps, 1),
        "loss_sca": loss_sca_sum / max(num_steps, 1),
        "loss_weight": float(loss_weight),
        "proto_pixels": [int(v) for v in proto_pixels],
        "sca_pixels": int(sca_pixels),
        "num_steps": int(num_steps),
        **_ema_decay_stats(ema_decays),
    }


def select_prototypical_aware_samples(model_t, loader, prototypes, in_args, device):
    model_mode = model_t.training
    model_t.eval()

    fg_classes = (1, 2)
    class_pixel_count = {1: 0, 2: 0}

    with torch.no_grad():
        for sample in loader:
            data = sample["image"].to(device, non_blocking=True)
            logits_t, feat_t = model_t(data)
            probs_t = torch.sigmoid(logits_t)
            pseudo_cls, _, _ = pseudo_class_from_prob2ch(probs_t, target_size=feat_t.shape[2:])
            for cls_id in fg_classes:
                class_pixel_count[cls_id] += int(torch.sum(pseudo_cls == cls_id).item())

    total_fg = float(class_pixel_count[1] + class_pixel_count[2])
    prior_alpha = max(0.0, float(1024.0))
    freq = {}
    omega = {}
    for cls_id in fg_classes:
        f_c = (float(class_pixel_count[cls_id]) + prior_alpha) / max(total_fg + prior_alpha * len(fg_classes), 1.0)
        f_c = max(f_c, 1e-6)
        freq[cls_id] = f_c
        omega[cls_id] = min(1.0 / np.sqrt(f_c), float(5.0))

    records = []
    with torch.no_grad():
        for sample in loader:
            data = sample["image"].to(device, non_blocking=True)
            names = [normalize_name(name) for name in sample["img_name"]]

            logits_t, feat_t = model_t(data)
            probs_t = torch.sigmoid(logits_t)
            pseudo_cls, _, _ = pseudo_class_from_prob2ch(probs_t, target_size=feat_t.shape[2:])
            feat_norm = F.normalize(feat_t, dim=1, p=2, eps=1e-12)

            bsz = data.size(0)
            ch = feat_norm.size(1)
            for b in range(bsz):
                z = feat_norm[b].permute(1, 2, 0).contiguous().view(-1, ch)
                cls_flat = pseudo_cls[b].contiguous().view(-1)
                class_uncertainty = {}
                class_center = {}

                for cls_id in fg_classes:
                    idx = torch.where(cls_flat == cls_id)[0]
                    if idx.numel() > 0:
                        if prototypes[cls_id] is not None:
                            proto = F.normalize(prototypes[cls_id].detach(), dim=0, p=2, eps=1e-12)
                            sim = torch.matmul(z[idx], proto)
                            u_vals = float(omega[cls_id]) * (1.0 - sim)
                        else:
                            u_vals = z.new_full((idx.numel(),), float(omega[cls_id]))
                        topk = min(int(in_args.hard_topk_pixels), int(idx.numel()))
                        top_u, top_pos = torch.topk(u_vals, k=topk, largest=True)
                        top_idx = idx[top_pos]
                        hard_center = torch.mean(z[top_idx], dim=0)
                        class_uncertainty[cls_id] = float(top_u.mean().item())
                    else:
                        if prototypes[cls_id] is not None:
                            hard_center = prototypes[cls_id].detach()
                        else:
                            hard_center = torch.mean(z, dim=0)
                        class_uncertainty[cls_id] = float(omega[cls_id])

                    class_center[cls_id] = F.normalize(hard_center, dim=0, p=2, eps=1e-12).detach().cpu()

                image_unc = float(np.mean([class_uncertainty[c] for c in fg_classes]))
                records.append(
                    {
                        "img_name": names[b],
                        "uncertainty": image_unc,
                        "uncertainty_cup": class_uncertainty[1],
                        "uncertainty_disc": class_uncertainty[2],
                        "center_1": class_center[1],
                        "center_2": class_center[2],
                    }
                )

    if len(records) == 0:
        if model_mode:
            model_t.train()
        return [], [], {"dataset_size": 0, "active_num": 0}

    num_samples = len(records)
    active_num = compute_active_num(num_samples, in_args)

    kernel = torch.zeros((num_samples, num_samples), dtype=torch.float32)
    for cls_id in fg_classes:
        center_key = "center_{}".format(cls_id)
        center_mat = torch.stack([records[i][center_key] for i in range(num_samples)], dim=0).float()
        center_mat = F.normalize(center_mat, dim=1, p=2, eps=1e-12)
        sim = torch.matmul(center_mat, center_mat.t())
        sim = 0.5 * (sim + 1.0)
        kernel += sim
    kernel = kernel / float(len(fg_classes))

    uncertainty_vec = torch.tensor([records[i]["uncertainty"] for i in range(num_samples)], dtype=torch.float32)
    total_kernel = torch.sum(kernel, dim=1)
    selected_kernel = torch.zeros(num_samples, dtype=torch.float32)
    selected_mask = torch.zeros(num_samples, dtype=torch.bool)

    selected_indices = []
    score_map = {}
    for _ in range(active_num):
        candidate_idx = torch.where(~selected_mask)[0]
        if candidate_idx.numel() == 0:
            break
        diversity_gain = total_kernel[candidate_idx] - selected_kernel[candidate_idx]
        acquisition = uncertainty_vec[candidate_idx] * diversity_gain
        best_local = int(torch.argmax(acquisition).item())
        best_idx = int(candidate_idx[best_local].item())

        selected_mask[best_idx] = True
        selected_indices.append(best_idx)
        score_map[best_idx] = {
            "diversity_gain": float(diversity_gain[best_local].item()),
            "acq_score": float(acquisition[best_local].item()),
        }
        selected_kernel = selected_kernel + kernel[:, best_idx]

    selected_names = [records[i]["img_name"] for i in selected_indices]
    rank_map = {idx: rank + 1 for rank, idx in enumerate(selected_indices)}

    score_rows = []
    for i, row in enumerate(records):
        sc = score_map.get(i, {"diversity_gain": 0.0, "acq_score": 0.0})
        score_rows.append(
            {
                "img_name": row["img_name"],
                "uncertainty": float(row["uncertainty"]),
                "uncertainty_cup": float(row["uncertainty_cup"]),
                "uncertainty_disc": float(row["uncertainty_disc"]),
                "diversity_gain": float(sc["diversity_gain"]),
                "acq_score": float(sc["acq_score"]),
                "selection_rank": int(rank_map.get(i, 0)),
                "selected": 1 if i in rank_map else 0,
            }
        )

    if model_mode:
        model_t.train()

    selection_stats = {
        "dataset_size": int(num_samples),
        "active_num": int(len(selected_indices)),
        "active_ratio": float(len(selected_indices)) / float(num_samples),
        "class_pixel_count": {int(k): int(v) for k, v in class_pixel_count.items()},
        "class_freq": {int(k): float(v) for k, v in freq.items()},
        "class_weight": {int(k): float(v) for k, v in omega.items()},
    }
    return selected_names, score_rows, selection_stats


def adapt_epoch_label_guided(
    model_t,
    model_s,
    optim,
    labeled_loader,
    unlabeled_loader,
    in_args,
    pred_bank,
    pseudo_prototypes,
    golden_anchors,
    margins,
    *,
    policy=None,
):
    policy = _resolve_policy(policy)
    device = next(model_s.parameters()).device
    num_l = len(labeled_loader)
    num_u = len(unlabeled_loader)
    num_steps = max(num_l, num_u)
    if num_steps <= 0:
        return {
            "loss_total": 0.0,
            "loss_sup": 0.0,
            "loss_unsup": 0.0,
            "loss_sca": 0.0,
            "loss_cal": 0.0,
            "loss_weight": 1.0,
            "cal_pixels": 0,
            "sca_pixels": 0,
            "golden_pixels": [0, 0, 0],
            "pseudo_pixels": [0, 0, 0],
            "num_steps": 0,
            **_ema_decay_stats(()),
        }

    loss_weight = compute_class_balance_weight(pred_bank, in_args)
    labeled_iter = iter(labeled_loader) if num_l > 0 else None
    unlabeled_iter = iter(unlabeled_loader) if num_u > 0 else None

    loss_total_sum = 0.0
    loss_sup_sum = 0.0
    loss_unsup_sum = 0.0
    loss_sca_sum = 0.0
    loss_cal_sum = 0.0
    cal_pixels = 0
    sca_pixels = 0
    golden_pixels = [0, 0, 0]
    pseudo_pixels = [0, 0, 0]
    ema_decays = []

    for _ in range(num_steps):
        optim.zero_grad(set_to_none=True)
        loss_sup = torch.zeros(1, device=device)
        loss_unsup = torch.zeros(1, device=device)
        loss_sca = torch.zeros(1, device=device)
        loss_cal = torch.zeros(1, device=device)

        if num_l > 0:
            (sample_w_l, sample_s_l), labeled_iter = _fetch_next(labeled_loader, labeled_iter)
            imgs_w_l = sample_w_l["image"].to(device, non_blocking=True)
            imgs_s_l = sample_s_l["image"].to(device, non_blocking=True)
            labels_l = sample_s_l["label"].to(device, non_blocking=True)
            names_l = [normalize_name(name) for name in sample_w_l["img_name"]]

            logits_s_l, feat_s_l = model_s(imgs_s_l)
            probs_s_l = torch.sigmoid(logits_s_l)
            loss_sup_bce = F.binary_cross_entropy_with_logits(logits_s_l, labels_l)
            loss_sup_dice = dice_loss_multichannel(probs_s_l, labels_l)
            loss_sup = loss_sup_bce + loss_sup_dice

            labels_low_l = F.interpolate(labels_l, size=feat_s_l.shape[2:], mode="nearest")
            gt_masks_l = build_exclusive_masks_from_gt(labels_low_l)
            if policy.golden_update_timing == "before_loss":
                count_g = update_prototypes_ema(golden_anchors, feat_s_l.detach(), gt_masks_l, 0.95)
                for cls_id in (0, 1, 2):
                    golden_pixels[cls_id] += count_g[cls_id]
            loss_cal, n_cal = policy.guided_cal(
                feat_s_l,
                gt_masks_l,
                golden_anchors,
                margins,
                in_args,
                max_pixels_per_class=1024,
            )
            cal_pixels += n_cal

            with torch.no_grad():
                logits_t_l, _ = model_t(imgs_w_l)
                probs_t_l = torch.sigmoid(logits_t_l)
                for idx, name in enumerate(names_l):
                    pred_bank[name] = probs_t_l[idx].detach().clone()
        if num_u > 0:
            (sample_w_u, sample_s_u), unlabeled_iter = _fetch_next(unlabeled_loader, unlabeled_iter)
            imgs_w_u = sample_w_u["image"].to(device, non_blocking=True)
            imgs_s_u = sample_s_u["image"].to(device, non_blocking=True)
            names_u = [normalize_name(name) for name in sample_w_u["img_name"]]

            logits_s_u, feat_s_u = model_s(imgs_s_u)
            probs_s_u = torch.sigmoid(logits_s_u)
            with torch.no_grad():
                logits_t_u, feat_t_u = model_t(imgs_w_u)
                probs_t_u = torch.sigmoid(logits_t_u)
            pseudo_u = soft_label_to_hard(probs_t_u, in_args.pseudo_label_threshold)
            loss_unsup_pixel = F.binary_cross_entropy_with_logits(logits_s_u, pseudo_u, reduction="none")
            loss_mask = torch.ones_like(pseudo_u)
            loss_mask[:, 0, ...][pseudo_u[:, 0, ...] == 0] = loss_weight
            loss_unsup_bce = torch.sum(loss_unsup_pixel * loss_mask) / torch.sum(loss_mask).clamp_min(1e-6)
            # Domain1's retained run used BCE only for calibration pseudo labels.
            loss_unsup = (loss_unsup_bce if in_args.dataset == "Domain1" and policy is REGRESSION_LAGGED else
                          loss_unsup_bce + dice_loss_multichannel(probs_s_u, pseudo_u))

            masks_u = build_exclusive_masks_from_probs(
                probs_t_u, conf_thr=0.995, target_size=feat_t_u.shape[2:]
            )
            count_u = update_prototypes_ema(pseudo_prototypes, feat_t_u.detach(), masks_u, 0.95)
            for cls_id in (0, 1, 2):
                pseudo_pixels[cls_id] += count_u[cls_id]

            pseudo_cls_s, conf_s, _ = pseudo_class_from_prob2ch(probs_t_u, target_size=feat_s_u.shape[2:])
            valid_s = conf_s >= 0.95
            ref_prototypes = get_reference_prototypes(golden_anchors, pseudo_prototypes)
            loss_sca, n_sca = policy.guided_sca(
                feat_s_u,
                pseudo_cls_s,
                ref_prototypes,
                in_args,
                valid_mask=valid_s,
                max_pixels_per_class=512,
            )
            sca_pixels += n_sca

            for idx, name in enumerate(names_u):
                pred_bank[name] = probs_t_u[idx].detach().clone()

        loss = loss_sup + loss_unsup + in_args.beta_sca * loss_sca + in_args.lambda_cal * loss_cal
        loss.backward()
        optim.step()

        if num_l > 0 and policy.golden_update_timing == "after_step":
            count_g = update_prototypes_ema(golden_anchors, feat_s_l.detach(), gt_masks_l, 0.95)
            for cls_id in (0, 1, 2):
                golden_pixels[cls_id] += count_g[cls_id]

        with torch.no_grad():
            for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
                param_t.data = param_t.data * in_args.model_ema_rate + param_s.data * (1.0 - in_args.model_ema_rate)
        ema_decays.append(float(in_args.model_ema_rate))

        loss_total_sum += float(loss.item())
        loss_sup_sum += float(loss_sup.item())
        loss_unsup_sum += float(loss_unsup.item())
        loss_sca_sum += float(loss_sca.item())
        loss_cal_sum += float(loss_cal.item())

    return {
        "loss_total": loss_total_sum / num_steps,
        "loss_sup": loss_sup_sum / num_steps,
        "loss_unsup": loss_unsup_sum / num_steps,
        "loss_sca": loss_sca_sum / num_steps,
        "loss_cal": loss_cal_sum / num_steps,
        "loss_weight": float(loss_weight),
        "cal_pixels": int(cal_pixels),
        "sca_pixels": int(sca_pixels),
        "golden_pixels": [int(v) for v in golden_pixels],
        "pseudo_pixels": [int(v) for v in pseudo_pixels],
        "num_steps": int(num_steps),
        **_ema_decay_stats(ema_decays),
    }


def evaluate(model, data_loader, device):
    """Evaluate without changing training modes, BN buffers, or training RNG streams."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = getattr(data_loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    try:
        model.eval()
        return _evaluate_inference(model, data_loader, device)
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if loader_rng is not None:
            generator.set_state(loader_rng)


def _evaluate_inference(model, data_loader, device):
    val_dice = {"cup": np.array([]), "disc": np.array([])}
    val_assd = {"cup": np.array([]), "disc": np.array([])}

    with torch.no_grad():
        for sample in data_loader:
            data = sample["image"].to(device, non_blocking=True)
            target_map = sample["label"]
            predictions, _ = model(data)

            dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
            val_dice["cup"] = np.append(val_dice["cup"], dice_cup)
            val_dice["disc"] = np.append(val_dice["disc"], dice_disc)

            assd = assd_compute(predictions, target_map)
            val_assd["cup"] = np.append(val_assd["cup"], assd[:, 0])
            val_assd["disc"] = np.append(val_assd["disc"], assd[:, 1])

    avg_dice = [0.0, 0.0]
    std_dice = [0.0, 0.0]
    avg_assd = [0.0, 0.0]
    std_assd = [0.0, 0.0]

    avg_dice[0] = np.mean(val_dice["cup"])
    avg_dice[1] = np.mean(val_dice["disc"])
    std_dice[0] = np.std(val_dice["cup"])
    std_dice[1] = np.std(val_dice["disc"])

    val_assd["cup"] = np.delete(val_assd["cup"], np.where(val_assd["cup"] == -1))
    val_assd["disc"] = np.delete(val_assd["disc"], np.where(val_assd["disc"] == -1))
    avg_assd[0] = np.mean(val_assd["cup"]) if val_assd["cup"].size > 0 else -1
    avg_assd[1] = np.mean(val_assd["disc"]) if val_assd["disc"].size > 0 else -1
    std_assd[0] = np.std(val_assd["cup"]) if val_assd["cup"].size > 0 else -1
    std_assd[1] = np.std(val_assd["disc"]) if val_assd["disc"].size > 0 else -1

    return avg_dice, std_dice, avg_assd, std_assd


def better_validation_checkpoint(row, best, warmup_epochs):
    """Only calibration epochs compete; exact ties retain the earliest observed epoch."""
    if row["epoch"] <= warmup_epochs or row["phase"] != "calibration":
        return False
    if not np.isfinite(row["avg_dice"]) or row["avg_dice"] < 0:
        raise FloatingPointError("validation mean Dice must be finite and nonnegative")
    if best is None or row["avg_dice"] > best["avg_dice"]:
        return True
    if row["avg_dice"] != best["avg_dice"]:
        return False
    current_assd = row["avg_assd"] if np.isfinite(row["avg_assd"]) and row["avg_assd"] >= 0 else float("inf")
    best_assd = best["avg_assd"] if np.isfinite(best["avg_assd"]) and best["avg_assd"] >= 0 else float("inf")
    return current_assd < best_assd


def teacher_ema_metadata(policy, args):
    return dict(rule="fixed", decay=args.model_ema_rate,
                update="teacher = decay * teacher + (1-decay) * updated student; parameters only")


def format_eval_log(prefix, avg_dice, std_dice, avg_assd, std_assd):
    valid_assd = [v for v in avg_assd if v >= 0]
    assd_mean = float(np.mean(valid_assd)) if len(valid_assd) > 0 else -1.0
    return (
        "{} dice: cup: {:.4f}+-{:.4f} disc: {:.4f}+-{:.4f} avg: {:.4f}, "
        "assd: cup: {:.4f}+-{:.4f} disc: {:.4f}+-{:.4f} avg: {:.4f}"
    ).format(
        prefix,
        avg_dice[0],
        std_dice[0],
        avg_dice[1],
        std_dice[1],
        (avg_dice[0] + avg_dice[1]) / 2.0,
        avg_assd[0],
        std_assd[0],
        avg_assd[1],
        std_assd[1],
        assd_mean,
    )


def calibration_epoch_lr(args, policy, epoch):
    """Domain2's retained cosine schedule; epochs 1–11 keep the original LR."""
    if (args.dataset != "Domain2" or policy is not REGRESSION_LAGGED
            or epoch <= args.warmup_epochs):
        return args.lr
    progress = (epoch - args.warmup_epochs) / (args.epoch - args.warmup_epochs)
    return args.lr * (1.0 + math.cos(math.pi * progress)) / 2.0


def main(args, *, policy=None):
    """Run a single-domain or compound/open experiment with one shared training path."""
    if policy is None:
        policy = REGRESSION_LAGGED if args.dataset in ("Domain1", "Domain2") else CLASSIFICATION
    policy = _resolve_policy(policy)
    if args.dataset == "Domain4" and policy is not CLASSIFICATION:
        raise ValueError("Domain4 requires the original CLASSIFICATION policy")
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weak = transforms.Compose([trans.Resize(512), trans.slight_color_shift(),
                               trans.Normalize_tf(), trans.ToTensor()])
    strong = transforms.Compose([trans.Resize(512), trans.add_salt_pepper_noise(),
                                 trans.eraser(), trans.adjust_light(), trans.adjust_contrast(),
                                 trans.Normalize_tf(), trans.ToTensor()])
    test_transform = transforms.Compose([trans.Resize(512), trans.Normalize_tf(), trans.ToTensor()])
    adapt_domains, compound_domains, open_domains = experiment_domains(args.dataset)
    common = dict(base_dir=str(resolve_path(args.data_dir)), dataset=adapt_domains,
                  split="train/ROIs", include_domain_prefix=True)
    train_set = fundus_dataloader2.FundusSegmentation_2transform(
        **common, transform_weak=weak, transform_strong=strong, labeled_names=set())
    weak_set = fundus_dataloader2.FundusSegmentation(**common, transform=weak, with_label=False)
    validation_set = None
    if args.validation_split:
        validation_set = fundus_dataloader2.FundusSegmentation(**common, transform=test_transform)
        apply_validation_split(args, train_set, weak_set, validation_set)
    for count in (len(train_set), args.active_num, len(train_set) - args.active_num):
        batch = min(args.batch_size, count)
        if batch < 2 or count % batch == 1:
            raise ValueError("BatchNorm requires >=2 images in every batch; adjust batch-size or split")
    test_sets = {} if args.skip_test else {
        domain: fundus_dataloader2.FundusSegmentation(str(resolve_path(args.data_dir)),
            dataset=domain, split="test/ROIs", transform=test_transform, include_domain_prefix=True)
        for domain in compound_domains + open_domains}
    def loader(dataset, shuffle=False, batch_size=None, evaluation=False):
        return DataLoader(dataset, batch_size=batch_size or args.batch_size,
                          shuffle=shuffle, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed) if evaluation else None)
    test_loaders = {domain: loader(dataset, evaluation=True) for domain, dataset in test_sets.items()}
    validation_loader = loader(validation_set, evaluation=True) if validation_set is not None else None
    train_loader, weak_loader = loader(train_set, True), loader(weak_set)
    output, metadata = create_run(args, train_set, test_sets, validation_set)
    policy_metadata = dict(
        method_variant=policy.method_variant,
        alignment_operators=dict(warmup_sca=policy.warmup_sca.__name__,
                                 guided_sca=policy.guided_sca.__name__,
                                 guided_cal=policy.guided_cal.__name__),
        golden_update_timing=policy.golden_update_timing,
        inactive_parameters=list(policy.inactive_parameters), ema_rule=policy.ema_rule,
        teacher_ema=teacher_ema_metadata(policy, args))
    policy_metadata["segmentation_loss_by_branch"] = dict(
        warmup="bce_dice", labeled="bce_dice",
        calibration_unlabeled="bce" if args.dataset == "Domain1" and policy is REGRESSION_LAGGED else "bce_dice")
    policy_metadata["learning_rate_schedule"] = (
        dict(rule="calibration_cosine", base_lr=args.lr,
             constant_through_epoch=args.warmup_epochs + 1,
             cosine_intervals=args.epoch - args.warmup_epochs,
             formula="lr(e)=base_lr*(1+cos(pi*(e-constant_through_epoch)/cosine_intervals))/2; e is one-based")
        if args.dataset == "Domain2" and policy is REGRESSION_LAGGED
        else dict(rule="constant", base_lr=args.lr))
    metadata.update(policy_metadata)
    metadata.update(training_epochs=args.epoch,
                    preferred_checkpoint="best_adaptation_student.pth.tar" if validation_loader is not None
                    else "after_adaptation_student.pth.tar", selected_epoch=None if validation_loader is not None else args.epoch)
    write_json(output / "run_metadata.json", metadata)
    with (output / "train.log").open("w", encoding="utf-8") as stream:
        def log(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()
        log(json.dumps(effective_config(args), ensure_ascii=False))
        log("Policy: " + json.dumps(policy_metadata, ensure_ascii=False))
        log(f"Adaptation: {adapt_domains}; C tests: {compound_domains}; O tests: {open_domains}")
        log(f"A2L: warmup_epochs={args.warmup_epochs}, calibration_epochs={args.epoch - args.warmup_epochs}")
        student = netd.DeepLab(num_classes=2, backbone="mobilenet", output_stride=16,
                              sync_bn=True, freeze_bn=False).to(device)
        teacher = netd.DeepLab(num_classes=2, backbone="mobilenet", output_stride=16,
                              sync_bn=True, freeze_bn=False).to(device)
        load_checkpoint(student, args.model_file)
        load_checkpoint(teacher, args.model_file)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.99))
        student.train()
        teacher.train()
        teacher.requires_grad_(False)
        pred_bank = init_pred_bank(teacher, weak_loader, device)
        pseudo_prototypes, golden_anchors = [None] * 3, [None] * 3
        class_hist = torch.ones(3, dtype=torch.float32, device=device)
        def evaluate_tests(prefix):
            rows = []
            for domain, test_loader in test_loaders.items():
                metrics = evaluate(student, test_loader, device)
                log(format_eval_log(f"{prefix} [{domain}]", *metrics))
                rows.append(metric_row(domain, metrics))
            return rows
        if validation_loader is not None:
            initial = evaluate(student, validation_loader, device)
            log(format_eval_log(f"source [{args.dataset}]", *initial))
            write_json(output / "source_metrics.json", metric_row(args.dataset, initial))
        elif test_loaders:
            initial_rows = evaluate_tests("source")
            write_json(output / "source_metrics.json", initial_rows[0] if len(initial_rows) == 1 else
                       evaluation_summary(initial_rows, compound_domains, open_domains))
        labeled_loader, unlabeled_loader = None, None
        history = []
        validation_history, best_validation = [], None
        last_validation_metrics = None
        for epoch in range(args.epoch):
            log(f"\nepoch {epoch + 1}/{args.epoch}:")
            epoch_lr = calibration_epoch_lr(args, policy, epoch)
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
            log(f"lr={epoch_lr:.12g}")
            if epoch < args.warmup_epochs:
                stats = adapt_epoch_warmup(teacher, student, optimizer, train_loader,
                                           args, pred_bank, pseudo_prototypes, policy=policy)
                phase = "warmup"
            else:
                if labeled_loader is None:
                    names, score_rows, selection = select_prototypical_aware_samples(
                        teacher, weak_loader, pseudo_prototypes, args, device)
                    selected = set(names)
                    if len(selected) != args.active_num:
                        raise RuntimeError("active selection did not satisfy the annotation budget")
                    train_set.labeled_names = selected
                    (output / "selected_samples_a2l.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
                    write_csv(output / "selection_scores_a2l.csv", score_rows)
                    indices = [i for i, row in enumerate(train_set.image_list) if row["img_name"] in selected]
                    other = [i for i, row in enumerate(train_set.image_list) if row["img_name"] not in selected]
                    labeled_loader = loader(Subset(train_set, indices), True, min(args.batch_size, len(indices)))
                    unlabeled_loader = loader(Subset(train_set, other), True, min(args.batch_size, len(other)))
                    counts = bootstrap_golden_anchors(student, labeled_loader, golden_anchors, device)
                    class_hist = estimate_labeled_class_hist(labeled_loader, device)
                    selection["selected_mask_sha256"] = {
                        row["img_name"]: sha256_file(row["label"]) for row in train_set.image_list
                        if row["img_name"] in selected}
                    write_json(output / "selection_metadata.json", selection)
                    log(f"active selection: selected={len(names)}/{len(train_set)}, golden_pixels={counts}")
                current_hist = estimate_labeled_class_hist(labeled_loader, device)
                class_hist = 0.98 * class_hist + 0.02 * current_hist
                margins = compute_class_adaptive_margins(class_hist, args.cal_margin_scale)
                stats = adapt_epoch_label_guided(teacher, student, optimizer, labeled_loader,
                    unlabeled_loader, args, pred_bank, pseudo_prototypes, golden_anchors, margins,
                    policy=policy)
                phase = "calibration"
            if not all(np.isfinite(value) for value in stats.values() if isinstance(value, (float, int))):
                raise FloatingPointError(f"nonfinite training statistic at epoch {epoch + 1}")
            log(f"phase={phase}: " + ", ".join(
                f"{key.removeprefix('loss_')}={value:.6f}" for key, value in stats.items()
                if key.startswith("loss_")))
            log(f"ema_rule={policy.ema_rule}: " + ", ".join(
                f"{key}={stats[key]:.6f}" for key in ("ema_decay_mean", "ema_decay_min", "ema_decay_max")
                if stats.get(key) is not None))
            stats["lr"] = epoch_lr
            history.append(dict(epoch=epoch + 1, phase=phase, **stats))
            write_json(output / "history.json", history)
            if validation_loader is not None:
                last_validation_metrics = evaluate(student, validation_loader, device)
                validation_row = dict(epoch=epoch + 1, phase=phase,
                                      eligible_for_selection=epoch + 1 > args.warmup_epochs,
                                      **metric_row(args.dataset, last_validation_metrics))
                validation_history.append(validation_row)
                write_json(output / "validation_history.json", validation_history)
                write_csv(output / "validation_history.csv", validation_history)
                log(format_eval_log(f"validation epoch {epoch + 1} [{args.dataset}]", *last_validation_metrics))
                if better_validation_checkpoint(validation_row, best_validation, args.warmup_epochs):
                    best_validation = validation_row.copy()
                    selection = dict(rule=metadata["selection_rule"], selected_epoch=epoch + 1,
                                     selected_phase=phase, eligible_epochs=[args.warmup_epochs + 1, args.epoch],
                                     validation_split=metadata["validation_split"], metrics=best_validation)
                    payload = checkpoint_payload(student, args, metadata, "student", epoch=epoch + 1)
                    payload.update(training_epochs=args.epoch, method_variant=policy.method_variant,
                                   policy=policy_metadata, model_selection=selection)
                    best_path = output / "best_adaptation_student.pth.tar"
                    torch.save(payload, best_path)
                    write_json(output / "best_validation_metrics.json", best_validation)
                    write_csv(output / "best_validation_metrics.csv", [best_validation])
                    write_json(output / "best_checkpoint.json", dict(
                        epoch=epoch + 1, phase=phase, training_epochs=args.epoch, training_completed=False,
                        checkpoint=best_path.name, sha256=sha256_file(best_path), rule=metadata["selection_rule"],
                        policy=policy_metadata, metrics=best_validation))
        for role, model in (("teacher", teacher), ("student", student)):
            torch.save(checkpoint_payload(model, args, metadata, role), output / f"after_adaptation_{role}.pth.tar")
        if test_loaders:
            rows = evaluate_tests("final student")
            summary = evaluation_summary(rows, compound_domains, open_domains)
            write_json(output / "metrics.json", rows[0] if len(rows) == 1 else summary)
            write_csv(output / "metrics.csv", rows)
            fields = ("cup_dice", "disc_dice", "avg_dice", "cup_assd", "disc_assd", "avg_assd")
            summary_rows = [dict(group="C" if row["domain"] in compound_domains else "O",
                                 domain=row["domain"], **{key: row[key] for key in fields}) for row in rows]
            for name in ("avg_c", "avg_o", "avg_all_domains", "avg_c_plus_o"):
                if summary[name] is not None:
                    summary_rows.append(dict(group="summary", domain=name, **summary[name]))
            write_csv(output / "compound_eval_summary.csv", summary_rows)
        if validation_loader is not None:
            metrics = last_validation_metrics
            log(format_eval_log(f"validation final student [{args.dataset}]", *metrics))
            write_json(output / "validation_metrics.json", metric_row(args.dataset, metrics))
            write_csv(output / "validation_metrics.csv", [metric_row(args.dataset, metrics)])
            if best_validation is None:
                raise RuntimeError("no eligible calibration-epoch validation checkpoint was recorded")
            best_receipt = dict(epoch=best_validation["epoch"], phase=best_validation["phase"],
                                training_epochs=args.epoch, training_completed=True,
                                checkpoint="best_adaptation_student.pth.tar",
                                sha256=sha256_file(output / "best_adaptation_student.pth.tar"),
                                rule=metadata["selection_rule"], policy=policy_metadata, metrics=best_validation)
            write_json(output / "best_checkpoint.json", best_receipt)
            metadata["selected_epoch"] = best_validation["epoch"]
            metadata["best_student_sha256"] = best_receipt["sha256"]
        metadata.update(status="completed", epoch=args.epoch, selected_samples=args.active_num,
                        test_evaluated=bool(test_loaders),
                        final_student_sha256=sha256_file(output / "after_adaptation_student.pth.tar"))
        write_json(output / "run_metadata.json", metadata)
        log(f"Results: {output}")
    return output


def metric_row(domain, metrics):
    dice, dice_std, assd, assd_std = metrics
    row = {"domain": domain}
    for prefix, means, stds in (("dice", dice, dice_std), ("assd", assd, assd_std)):
        for index, name in enumerate(("cup", "disc")):
            row[f"{name}_{prefix}"] = float(means[index])
            row[f"{name}_{prefix}_std"] = float(stds[index])
        valid = [float(value) for value in means if value >= 0]
        row[f"avg_{prefix}"] = float(np.mean(valid)) if valid else -1.0
    return row


def aggregate_metrics(rows):
    """Equal domain weights, preserving the original invalid-ASSD handling."""
    if not rows:
        return None
    result = {}
    for metric in ("dice", "assd"):
        for region in ("cup", "disc"):
            key = f"{region}_{metric}"
            values = [row[key] for row in rows if metric == "dice" or row[key] >= 0]
            result[key] = float(np.mean(values)) if values else -1.0
        values = [result[f"{region}_{metric}"] for region in ("cup", "disc")
                  if result[f"{region}_{metric}"] >= 0]
        result[f"avg_{metric}"] = float(np.mean(values)) if values else -1.0
    return result


def evaluation_summary(rows, compound_domains, open_domains):
    avg_c = aggregate_metrics([row for row in rows if row["domain"] in compound_domains])
    avg_o = aggregate_metrics([row for row in rows if row["domain"] in open_domains])
    return dict(domains=rows, avg_c=avg_c, avg_o=avg_o,
                avg_all_domains=aggregate_metrics(rows),
                avg_c_plus_o={key: (avg_c[key] + avg_o[key]) / 2 if min(avg_c[key], avg_o[key]) >= 0 else -1.0
                              for key in avg_c} if avg_c and avg_o else None,
                aggregation="avg_all_domains: legacy domain-equal; avg_c_plus_o: (Avg.C + Avg.O)/2")


if __name__ == "__main__":
    args = parse_args()
    if args.print_config:
        print(json.dumps(effective_config(args), indent=2))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        load_runtime_dependencies()
        main(args)
