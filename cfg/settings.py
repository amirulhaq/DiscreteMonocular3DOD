import os
from utils.cs_dataset import T

from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root folders of cityscapes dataset
_C.DATASETS.ROOT = 'C:/Users/lps3090/Documents/Datasets/Cityscape'
_C.DATASETS.DETECT_CLASSES = ("Car",)

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0
# Normalization for backbone
_C.MODEL.BACKBONE.USE_NORMALIZATION = "GN"
_C.MODEL.BACKBONE.DOWN_RATIO = 4
_C.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = 32


# ---------------------------------------------------------------------------- #
# Heatmap Head options
# ---------------------------------------------------------------------------- #

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.USE_NORMALIZATION = "GN"
_C.MODEL.HEAD.NUM_CHANNEL = 256
# Loss weight for hm and reg loss
_C.MODEL.HEAD.GAUSSIAN_ROT = True
_C.MODEL.HEAD.GAUSSIAN_DIM = False
# Set depth as classification = True. If set to false, then the depth output will have 1 channel (regression)
_C.MODEL.HEAD.DEPTH_GAUSSIAN = True
_C.MODEL.HEAD.WITH_CENTER_REGRESSION = True
_C.MODEL.HEAD.DEPTH_BIN = 120

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCH_SIZE = 2
_C.TRAINING.LEARNING_RATE = 5e-3
_C.TRAINING.MOMENTUM = 0.937
_C.TRAINING.CONTINUE = True
_C.TRAINING.USE_GPU = True
_C.TRAINING.SAVE_EVERY_EPOCH = True
_C.TRAINING.SAVE_NUM_EPOCH = 5 # Define the number of epoch's interval to save the checkpoint
_C.TRAINING.DEBUG = False # If true the training will be a bit slower
_C.TRAINING.LR_REDUCE_INTERVAL = 10 # Define the number of epoch's interval for reducig learning rate
_C.TRAINING.LR_REDUCE_VALUE = 5 # new_lr = initial_lr / (LR_REDUCE_VALUE**(epoch//LR_REDUCE_INTERVAL))
# Weight for depth, orientationbin, orientationres, instance, hm, dim, dis losses, respectively
_C.TRAINING.WEIGHT = (0.05, 0.1, 0.25, 0.001, 0.05, 10, 0.05) #(0.05, 0.1, 0.25, 0.001, 0.1, 10, 0.05) 

_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 8
_C.EVAL.CONF_THRES = 0.01
_C.EVAL.WEIGHTS = 'checkpoint/model_final.pth'
_C.EVAL.SAVE_FOLDER = 'saves/predBbox3d/'
_C.EVAL.VISUALIZE_RESULTS = True
_C.EVAL.USE_CENTER_DISPARITY = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()