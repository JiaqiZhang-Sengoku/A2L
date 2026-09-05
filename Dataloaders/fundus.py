"""Fundus datasets for single-domain and joint-domain A2L experiments."""

import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _collect_images(image_dir):
    """Return the same deterministic, lexicographically sorted image order as before."""
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(glob(os.path.join(image_dir, pattern)))
    if not image_paths:
        raise FileNotFoundError("No supported images found in: {}".format(image_dir))
    return sorted(image_paths)


def _build_image_name(image_path, dataset, include_domain_prefix):
    base_name = os.path.basename(image_path)
    if include_domain_prefix:
        return "{}__{}".format(dataset, base_name)
    return base_name


def _build_mask_path(image_path):
    return image_path.replace(
        "{}image{}".format(os.sep, os.sep),
        "{}mask{}".format(os.sep, os.sep),
    )


def _load_target(label_path, image_size, load_label):
    if not load_label:
        return Image.new("L", image_size, color=255)
    if not os.path.isfile(label_path):
        raise FileNotFoundError("Required segmentation mask is missing: {}".format(label_path))
    target = Image.open(label_path)
    if target.mode == "RGB":
        target = target.convert("L")
    return target


class _FundusDatasetBase(Dataset):
    def __init__(
        self,
        base_dir,
        dataset,
        split,
        with_label,
        include_domain_prefix,
        labeled_names,
    ):
        self._base_dir = base_dir
        self.split = split
        self.dataset = dataset
        self.with_label = with_label
        self.include_domain_prefix = include_domain_prefix
        # None preserves the historical behavior (decode every mask). An empty
        # set makes every sample unlabeled; a populated set decodes only those
        # selected samples. The attribute intentionally remains assignable so
        # active selection can update it without rebuilding the dataset.
        self.labeled_names = None if labeled_names is None else set(labeled_names)

        self.image_list = []
        if isinstance(dataset, str):
            domains = [dataset]
        elif isinstance(dataset, (list, tuple)) and dataset and all(
            isinstance(domain, str) and domain for domain in dataset
        ):
            domains = list(dataset)
        else:
            raise TypeError("dataset must be a domain name or a non-empty list/tuple of domain names")

        for domain in domains:
            image_dir = os.path.join(self._base_dir, domain, split, "image")
            print(image_dir)
            for image_path in _collect_images(image_dir):
                self.image_list.append(
                    {
                        "image": image_path,
                        "label": _build_mask_path(image_path),
                        "img_name": _build_image_name(image_path, domain, include_domain_prefix),
                    }
                )
        print("Number of images in {}: {:d}".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def _sample(self, index):
        item = self.image_list[index]
        image = Image.open(item["image"]).convert("RGB")
        image_name = item["img_name"]
        load_label = self.with_label and (
            self.labeled_names is None or image_name in self.labeled_names
        )
        target = _load_target(item["label"], image.size, load_label=load_label)
        return {"image": image, "label": target, "img_name": image_name}

    def __str__(self):
        return "Fundus(split=" + str(self.split) + ")"


class FundusSegmentation(_FundusDatasetBase):
    def __init__(
        self,
        base_dir,
        dataset="refuge",
        split="train",
        transform=None,
        with_label=True,
        include_domain_prefix=False,
        labeled_names=None,
    ):
        super().__init__(
            base_dir=base_dir,
            dataset=dataset,
            split=split,
            with_label=with_label,
            include_domain_prefix=include_domain_prefix,
            labeled_names=labeled_names,
        )
        self.transform = transform

    def __getitem__(self, index):
        sample = self._sample(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class FundusSegmentation_2transform(_FundusDatasetBase):
    def __init__(
        self,
        base_dir,
        dataset="refuge",
        split="train",
        transform_weak=None,
        transform_strong=None,
        with_label=True,
        include_domain_prefix=False,
        labeled_names=None,
    ):
        super().__init__(
            base_dir=base_dir,
            dataset=dataset,
            split=split,
            with_label=with_label,
            include_domain_prefix=include_domain_prefix,
            labeled_names=labeled_names,
        )
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __getitem__(self, index):
        sample = self._sample(index)
        # Preserve the historical weak-then-strong call order because it fixes
        # random-number consumption and therefore the realized augmentations.
        weak_sample = self.transform_weak(sample)
        strong_sample = self.transform_strong(sample)
        return weak_sample, strong_sample
