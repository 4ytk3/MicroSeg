from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def list_image_mask_files(image_dir: Path, mask_dir: Path) -> Tuple[List[str], List[str]]:
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    if len(image_files) != len(mask_files):
        raise ValueError("Image/mask file count mismatch")
    if not image_files:
        raise ValueError("No image files found")
    return image_files, mask_files


def split_indices(
    num_items: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if train_split < 0 or val_split < 0 or test_split < 0:
        raise ValueError("Split ratios must be non-negative")
    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0):
        raise ValueError("train/val/test split ratios must sum to 1.0")

    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_count = int(num_items * train_split)
    val_count = int(num_items * val_split)
    test_count = int(num_items * test_split)
    remainder = num_items - (train_count + val_count + test_count)
    train_count += remainder

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def make_kfold_splits(num_items: int, k_fold: int, seed: int) -> List[List[int]]:
    if k_fold < 2:
        raise ValueError("k_fold must be at least 2")
    if num_items < k_fold:
        raise ValueError("k_fold cannot exceed number of items")
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_sizes = [num_items // k_fold] * k_fold
    for i in range(num_items % k_fold):
        fold_sizes[i] += 1
    folds: List[List[int]] = []
    cursor = 0
    for size in fold_sizes:
        folds.append(indices[cursor : cursor + size])
        cursor += size
    return folds


def split_train_val_indices(
    indices: List[int],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if val_split < 0 or val_split >= 1:
        raise ValueError("val_split must be in [0, 1)")
    if not indices:
        return [], []
    shuffled = list(indices)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = int(len(shuffled) * val_split)
    val_idx = shuffled[:val_count]
    train_idx = shuffled[val_count:]
    return train_idx, val_idx


def _mask_center(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = mask.shape
        return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def _choose_negative_point(
    positive_mask: np.ndarray,
    ring_width: int,
    background_mask: np.ndarray | None = None,
) -> Tuple[int, int] | None:
    height, width = positive_mask.shape
    if background_mask is None:
        background = positive_mask == 0
    else:
        if background_mask.shape != positive_mask.shape:
            raise ValueError(
                f"background_mask shape mismatch: {background_mask.shape} vs {positive_mask.shape}"
            )
        background = background_mask.astype(bool)

    if ring_width > 0:
        ys, xs = np.where(positive_mask > 0)
        if ys.size > 0 and xs.size > 0:
            x0 = max(int(xs.min()) - ring_width, 0)
            x1 = min(int(xs.max()) + ring_width, width - 1)
            y0 = max(int(ys.min()) - ring_width, 0)
            y1 = min(int(ys.max()) + ring_width, height - 1)
            for _ in range(20):
                x = random.randint(x0, x1)
                y = random.randint(y0, y1)
                if background[y, x]:
                    return x, y
    ys, xs = np.where(background)
    if ys.size == 0 or xs.size == 0:
        return None
    idx = random.randrange(ys.size)
    return int(xs[idx]), int(ys[idx])


class InstanceMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_files: List[str],
        mask_files: List[str],
        indices: List[int],
        transform=None,
        max_instances_per_image: int = 10,
        negative_sample_prob: float = 0.0,
        ring_width: int = 0,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.mask_files = mask_files
        self.indices = indices
        self.transform = transform
        self.max_instances_per_image = int(max_instances_per_image)
        self.negative_sample_prob = float(negative_sample_prob)
        self.ring_width = int(ring_width)

    def __len__(self) -> int:
        return len(self.indices) * self.max_instances_per_image

    def __getitem__(self, idx: int):
        img_idx = self.indices[idx // self.max_instances_per_image]
        img_path = self.image_dir / self.image_files[img_idx]
        msk_path = self.mask_dir / self.mask_files[img_idx]

        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(msk_path).convert("L")
        mask_np = np.array(mask_img)

        ids = np.unique(mask_np)
        ids = ids[ids != 0]
        if len(ids) == 0:
            raise ValueError(f"No instances in {msk_path}")
        inst_id = int(random.choice(ids))

        bin_mask = (mask_np == inst_id).astype(np.uint8)
        cx, cy = _mask_center(bin_mask)

        points = [(cx, cy)]
        labels = [1]
        if self.negative_sample_prob > 0 and random.random() < self.negative_sample_prob:
            neg = _choose_negative_point(
                bin_mask,
                self.ring_width,
                background_mask=(mask_np == 0),
            )
            if neg is not None:
                points.append(neg)
                labels.append(0)

        input_point = torch.tensor([points], dtype=torch.float)
        input_label = torch.tensor([labels], dtype=torch.int)

        if self.transform:
            image = self.transform(image)

        bin_mask_t = torch.tensor(bin_mask, dtype=torch.float).unsqueeze(0)

        return {
            "image": image,
            "mask": bin_mask_t,
            "input_point": input_point,
            "input_label": input_label,
        }


class ExhaustiveInstanceMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_files: List[str],
        mask_files: List[str],
        indices: List[int],
        transform=None,
        negative_sample_prob: float = 0.0,
        ring_width: int = 0,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.mask_files = mask_files
        self.indices = indices
        self.transform = transform
        self.negative_sample_prob = float(negative_sample_prob)
        self.ring_width = int(ring_width)

        samples: List[Tuple[int, int]] = []
        for img_idx in indices:
            msk_path = self.mask_dir / self.mask_files[img_idx]
            mask_img = Image.open(msk_path).convert("L")
            mask_np = np.array(mask_img)
            ids = np.unique(mask_np)
            ids = ids[ids != 0]
            for inst_id in ids.tolist():
                samples.append((img_idx, int(inst_id)))
        if not samples:
            raise ValueError("No instances found for exhaustive instance sampling")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_idx, inst_id = self.samples[idx]
        img_path = self.image_dir / self.image_files[img_idx]
        msk_path = self.mask_dir / self.mask_files[img_idx]

        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(msk_path).convert("L")
        mask_np = np.array(mask_img)

        bin_mask = (mask_np == inst_id).astype(np.uint8)
        if bin_mask.sum() == 0:
            raise ValueError(f"Instance id {inst_id} not found in {msk_path}")
        cx, cy = _mask_center(bin_mask)

        points = [(cx, cy)]
        labels = [1]
        if self.negative_sample_prob > 0 and random.random() < self.negative_sample_prob:
            neg = _choose_negative_point(
                bin_mask,
                self.ring_width,
                background_mask=(mask_np == 0),
            )
            if neg is not None:
                points.append(neg)
                labels.append(0)

        input_point = torch.tensor([points], dtype=torch.float)
        input_label = torch.tensor([labels], dtype=torch.int)

        if self.transform:
            image = self.transform(image)

        bin_mask_t = torch.tensor(bin_mask, dtype=torch.float).unsqueeze(0)

        return {
            "image": image,
            "mask": bin_mask_t,
            "input_point": input_point,
            "input_label": input_label,
        }


class FullImageMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_files: List[str],
        mask_files: List[str],
        indices: List[int],
        transform=None,
        negative_sample_prob: float = 0.0,
        ring_width: int = 0,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.mask_files = mask_files
        self.indices = indices
        self.transform = transform
        self.negative_sample_prob = float(negative_sample_prob)
        self.ring_width = int(ring_width)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img_idx = self.indices[idx]
        img_path = self.image_dir / self.image_files[img_idx]
        msk_path = self.mask_dir / self.mask_files[img_idx]

        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(msk_path).convert("L")
        mask_np = np.array(mask_img)

        bin_mask = (mask_np > 0).astype(np.uint8)
        cx, cy = _mask_center(bin_mask)

        points = [(cx, cy)]
        labels = [1]
        if self.negative_sample_prob > 0 and random.random() < self.negative_sample_prob:
            neg = _choose_negative_point(
                bin_mask,
                self.ring_width,
                background_mask=(mask_np == 0),
            )
            if neg is not None:
                points.append(neg)
                labels.append(0)

        input_point = torch.tensor([points], dtype=torch.float)
        input_label = torch.tensor([labels], dtype=torch.int)

        if self.transform:
            image = self.transform(image)

        bin_mask_t = torch.tensor(bin_mask, dtype=torch.float).unsqueeze(0)

        return {
            "image": image,
            "mask": bin_mask_t,
            "input_point": input_point,
            "input_label": input_label,
        }


def create_dataloader(
    args: argparse.Namespace,
    image_dir: Path,
    mask_dir: Path,
    image_files: List[str],
    mask_files: List[str],
    indices: List[int],
    shuffle: bool,
) -> DataLoader:
    if args.sample_mode == "image":
        dataset: Dataset = FullImageMaskDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_files=image_files,
            mask_files=mask_files,
            indices=indices,
            negative_sample_prob=args.negative_prob,
            ring_width=args.ring_width,
        )
    elif args.sample_mode == "instance_all":
        dataset = ExhaustiveInstanceMaskDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_files=image_files,
            mask_files=mask_files,
            indices=indices,
            negative_sample_prob=args.negative_prob,
            ring_width=args.ring_width,
        )
    else:
        dataset = InstanceMaskDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_files=image_files,
            mask_files=mask_files,
            indices=indices,
            max_instances_per_image=args.instances_per_image,
            negative_sample_prob=args.negative_prob,
            ring_width=args.ring_width,
        )

    def seed_worker(worker_id: int) -> None:
        if args.seed is None:
            return
        worker_seed = int(args.seed) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(int(args.seed))

    batch_size = args.batch_size if args.backend == "hf" else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda b: b,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )
