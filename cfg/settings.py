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
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# train split tor dataset
_C.DATASETS.TRAIN_SPLIT = ""
# test split for dataset
_C.DATASETS.TEST_SPLIT = ""
_C.DATASETS.DETECT_CLASSES = ("Car",)
_C.DATASETS.MAX_OBJECTS = 30
_C.DATASETS.USE_CACHE = True
_C.DATASETS.ROOT = 'D:/Cityscape'

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
_C.MODEL.BACKBONE.CONV_BODY = "DLA-34-DCN"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0
# Normalization for backbone
_C.MODEL.BACKBONE.USE_NORMALIZATION = "GN"
_C.MODEL.BACKBONE.DOWN_RATIO = 4
_C.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = 32

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# _C.MODEL.HEAD.GAUSSIAN_DEPTH = True
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.GAUSSIAN_ROT = True
_C.MODEL.HEAD.GAUSSIAN_DIM = False
_C.MODEL.HEAD.DEPTH_GAUSSIAN = True
_C.MODEL.HEAD.DEPTH_GAUSSIAN_SHAPE = 3 # default = 3 if _C.MODEL.HEAD.DEPTH_GAUSSIAN = False then this doesn't matter
_C.MODEL.HEAD.WITH_CENTER_REGRESSION = True
_C.MODEL.HEAD.DEPTH_BIN = 120


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 25
_C.TRAINING.BATCH_SIZE = 2
_C.TRAINING.LEARNING_RATE = 5e-3
_C.TRAINING.MOMENTUM = 0.937
_C.TRAINING.CONTINUE = False
_C.TRAINING.USE_GPU = True
_C.TRAINING.SAVE_EVERY_EPOCH = True
_C.TRAINING.SAVE_NUM_EPOCH = 5 # Define the number of epoch's interval to save the checkpoint
_C.TRAINING.DEBUG = False # If true the training will be a bit slower
_C.TRAINING.LR_REDUCE_INTERVAL = 10 # Define the number of epoch's interval for reducig learning rate
_C.TRAINING.LR_REDUCE_VALUE = 5 # new_lr = initial_lr / (LR_REDUCE_VALUE**(epoch//LR_REDUCE_INTERVAL))
_C.TRAINING.FILE_FORMAT = 'final' #'depth_hard_label' #'depth_as_regression'#'hopefullyfinal'
# Weight for depth, orientationbin, orientationres, instance, hm, dim, dis losses, respectively
_C.TRAINING.WEIGHT = [0.05, 0.1, 0.25, 0.001, 0.25, 10, 0.05] 
_C.TRAINING.WARMUP = True

_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 4
_C.EVAL.CONF_THRES = 0.01
_C.EVAL.FILE_FORMAT = 'final/'
_C.EVAL.SAVE_FOLDER = 'saves/final/' 
_C.EVAL.VISUALIZE_RESULTS = True
_C.EVAL.USE_CENTER_DISPARITY = False
_C.EVAL.MODEL = 'checkpoint/model_final.pth' 
_C.EVAL.DEMO_VIDEO = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()