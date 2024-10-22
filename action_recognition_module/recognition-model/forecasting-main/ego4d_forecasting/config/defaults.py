#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "Kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# If True, adds final activation to model when evaluating
_C.TEST.NO_ACT = True

_C.TEST.EVAL_VAL = False



# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slow"

# Model name
_C.MODEL.MODEL_NAME = "ResNet"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [400]

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Size of feature for each input clip (right three dims will be pooled with all
# other input clips).
_C.MODEL.MULTI_INPUT_FEATURES = 2048

# Transformer number of heads.
_C.MODEL.TRANSFORMER_HEADS = 8

# Transformer depth.
_C.MODEL.TRANSFORMER_LAYERS = 6

_C.MODEL.TRANSFORMER_FROM_PRETRAIN = True

# Path to the bert embeddings of the possible noun labels in the Ego4D dataset 
_C.MODEL.BERT_EMBEDS_PATH = ""


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# Model head path if any
_C.DATA.CHECKPOINT_MODULE_FILE_PATH = "ego4d/models/"

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# If True, calculate the map as metric.
_C.DATA.TASK = "single-label"



# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# Gradually warm up the SOLVER.BASE_LR over this number of steps.
_C.SOLVER.WARMUP_STEPS = 1000

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Which PyTorch Lightning accelerator to use
_C.SOLVER.ACCELERATOR = "ddp"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Whether to enable logging
_C.ENABLE_LOGGING = True

# Log gradient distributions at period. Don't log if None.
_C.LOG_GRADIENT_PERIOD = -1

# Run 1 train, val and test batch for debuggin
_C.FAST_DEV_RUN = False

# Path to the checkpoint to load the initial weight.
_C.CHECKPOINT_FILE_PATH = ""

# Whether the checkpoint follows the caffe2 format
_C.CHECKPOINT_VERSION = ""

# Whether to load model head or not. Useful for loading pretrained models.
_C.CHECKPOINT_LOAD_MODEL_HEAD = False

# Whether or not to run on fblearner
_C.FBLEARNER = False


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True



# -----------------------------------------------------------------------------
# Forecasting options (LTA + STA)
# -----------------------------------------------------------------------------

_C.FORECASTING = CfgNode()
# _C.FORECASTING.BACKBONE = "SlowFast" # _C.MODEL.ARCH also has this info

# The number of future actions to return from the Epic Kitchen forecasting dataset.
_C.FORECASTING.NUM_ACTIONS_TO_PREDICT = 1
# TODO: LTA: _C.FORECASTING.NUM_ACTIONS_TO_PREDICT = 20

# The number of future action sequences to predict.
_C.FORECASTING.NUM_SEQUENCES_TO_PREDICT = 5

# Number of input clips before the chosen action (only supported by forecasting)
_C.FORECASTING.NUM_INPUT_CLIPS = 1



def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
