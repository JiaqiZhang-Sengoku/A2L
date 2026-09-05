"""Transforms used by the Domain1/Domain2 A2L adaptation pipeline."""

import random

import cv2
import numpy as np
import torch
from PIL import Image


def to_multilabel(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask


class add_salt_pepper_noise:
    def __call__(self, sample):
        image = np.array(sample["image"]).astype(np.uint8)
        image_copy = image.copy()
        salt_vs_pepper = 0.2
        amount = 0.004

        num_salt = np.ceil(amount * image_copy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * image_copy.size * (1.0 - salt_vs_pepper))

        seed = random.random()
        if seed > 0.75:
            coords = [np.random.randint(0, size - 1, int(num_salt)) for size in image_copy.shape]
            # Keep the historical value of 1 (not 255) for exact stage-A behavior.
            image_copy[coords[0], coords[1], :] = 1
        elif seed > 0.5:
            coords = [np.random.randint(0, size - 1, int(num_pepper)) for size in image_copy.shape]
            image_copy[coords[0], coords[1], :] = 0

        return {
            "image": image_copy,
            "label": sample["label"],
            "img_name": sample["img_name"],
        }


class adjust_light:
    def __call__(self, sample):
        image = sample["image"]
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((value / 255.0) ** inv_gamma) * 255 for value in np.arange(0, 256)]
            ).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
            return {
                "image": image,
                "label": sample["label"],
                "img_name": sample["img_name"],
            }
        return sample


class adjust_contrast:
    def __call__(self, sample):
        image = np.array(sample["image"]).astype(np.uint8)
        seed = random.random()
        if seed > 0.5:
            factor = random.uniform(0.6, 1.4)
            image = np.clip(
                (image.astype(np.float32) - 127.5) * factor + 127.5,
                0,
                255,
            ).astype(np.uint8)
            return {
                "image": image,
                "label": sample["label"],
                "img_name": sample["img_name"],
            }
        return sample


class slight_color_shift:
    def __call__(self, sample):
        image = np.array(sample["image"]).astype(np.uint8)
        seed = random.random()
        if seed > 0.5:
            channel_shift = np.random.uniform(-8.0, 8.0, size=(1, 1, 3))
            image = np.clip(image.astype(np.float32) + channel_shift, 0, 255).astype(np.uint8)
            return {
                "image": image,
                "label": sample["label"],
                "img_name": sample["img_name"],
            }
        return sample


class eraser:
    def __call__(
        self,
        sample,
        s_l=0.02,
        s_h=0.06,
        r_1=0.3,
        r_2=0.6,
        v_l=0,
        v_h=255,
        pixel_level=False,
    ):
        image = sample["image"]
        img_h, img_w, img_c = image.shape

        if random.random() > 0.5:
            return sample

        while True:
            area = np.random.uniform(s_l, s_h) * img_h * img_w
            ratio = np.random.uniform(r_1, r_2)
            width = int(np.sqrt(area / ratio))
            height = int(np.sqrt(area * ratio))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + width <= img_w and top + height <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(v_l, v_h, (height, width, img_c))
        else:
            value = np.random.uniform(v_l, v_h)
        image[top : top + height, left : left + width, :] = value

        return {
            "image": image,
            "label": sample["label"],
            "img_name": sample["img_name"],
        }


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]
        assert image.width == mask.width
        assert image.height == mask.height

        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return {"image": image, "label": mask, "img_name": sample["img_name"]}


class Normalize_tf:
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        # Retain the constructor API even though the historical implementation
        # always maps RGB from [0, 255] to [-1, 1].
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = np.array(sample["image"]).astype(np.float32)
        raw_mask = np.array(sample["label"]).astype(np.uint8)
        image /= 127.5
        image -= 1.0

        remapped = np.zeros([raw_mask.shape[0], raw_mask.shape[1]])
        remapped[raw_mask > 200] = 255
        remapped[(raw_mask > 50) & (raw_mask < 201)] = 128
        raw_mask[remapped == 0] = 2
        raw_mask[remapped == 255] = 0
        raw_mask[remapped == 128] = 1

        return {
            "image": image,
            "label": to_multilabel(raw_mask),
            "img_name": sample["img_name"],
        }


class ToTensor:
    def __call__(self, sample):
        image = np.array(sample["image"]).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(sample["label"]).astype(np.uint8).transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(mask).float(),
            "img_name": sample["img_name"],
        }
