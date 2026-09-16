#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal helper."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from core.registry import DATASETS
from data.lingbotmap.lerobot_v3 import LeRobotV3Dataset
from data.x2robot.mint_dataset import MyDataDataset
from data.sampling import (
    dataset_sample_seed,
    deterministic_subset_indices,
    retained_sample_count,
    validate_sample_fraction,
)
from utils.logging import rank0_print


def _dataset_name(root) -> str:
    path = Path(str(root))
    return path.parent.name if path.name == "lerobot_v3" else path.name


def _build_child_dataset(child_cfg: dict):
    root = child_cfg["root"]
    name = _dataset_name(root)
    fraction = validate_sample_fraction(
        child_cfg.get("sample_fraction", 1.0),
        context=f"data.root[{name}].sample_fraction",
    )
    if fraction == 0.0:
        rank0_print(f"[data-sampling] {name}: sample_fraction=0, skipped")
        return None

    dataset = DATASETS.build_from_cfg(child_cfg)
    total = len(dataset)
    keep = retained_sample_count(total, fraction)
    if keep < total:
        base_seed = child_cfg.get("sample_seed", 0)
        seed = dataset_sample_seed(base_seed, str(Path(str(root)).resolve()))
        dataset = Subset(dataset, deterministic_subset_indices(total, keep, seed))
    rank0_print(
        f"[data-sampling] {name}: sample_fraction={fraction:.6g}, "
        f"selected={keep:,}/{total:,} clips"
    )
    return dataset


def build_dataset(cfg: dict):
    root = cfg["root"]
    if isinstance(root, list):
        children = []
        for item in root:
            if isinstance(item, str):
                child_cfg = {**cfg, "root": item}
            elif isinstance(item, dict):
                if "root" not in item:
                    raise KeyError("[train]")
                if "camera_translation_normalization" in item:
                    raise KeyError("[train]")
                child_cfg = {**cfg, **item}
            else:
                raise TypeError(
                    f"[train]  {type(item).__name__}."
                )
            child = _build_child_dataset(child_cfg)
            if child is not None:
                children.append(child)
        if not children:
            raise ValueError("[train]")
        return ConcatDataset(children)
    dataset = _build_child_dataset(cfg)
    if dataset is None:
        raise ValueError("data.sample_fraction=0 leaves no training data")
    return dataset


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(cfg: dict, dataset=None, seed=None):
    """Internal helper."""
    ds = dataset if dataset is not None else build_dataset(cfg)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=bool(cfg.get("shuffle", True)),
        num_workers=int(cfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
