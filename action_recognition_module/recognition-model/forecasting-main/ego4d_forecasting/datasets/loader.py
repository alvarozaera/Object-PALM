#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler

from .build import build_dataset


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = False
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training

    sampler = None
    if not cfg.FBLEARNER:
        # Create a sampler for multi-process training
        if hasattr(dataset, "sampler"):
            sampler = dataset.sampler
        elif cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
    )
    return loader

