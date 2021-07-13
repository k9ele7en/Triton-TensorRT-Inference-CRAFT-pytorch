# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file configs to convert multiple model format for Triton
_____________________________________________________________________________
"""
from yacs.config import CfgNode as CN

_C = CN()

##################
#################
_C._BASE_ = CN()

###################
###################
# ---------------------------------------------------------------------------- #
# INFERENCE
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()
# threshold for texts heat map, 0 < text_threshold < 1, smaller means fewer text filtered, float
_C.INFERENCE.TEXT_THRESHOLD = 0.7

# threshold for affinities heat map between characters, 0 < link_threshold < 1, bigger means fewer link filtered, float
_C.INFERENCE.LINK_THRESHOLD = 0.4

# threshold for accept a word to be detected, 0 < text_threshold < 1, smaller means fewer text filtered, float
_C.INFERENCE.LOW_TEXT_SCORE = 0.4

_C.INFERENCE.OX_DO_CONSTANT_FOLDING = True

_C.INFERENCE.OX_EXPORT_PARAMS = True

_C.INFERENCE.OX_OPSET = 11

_C.INFERENCE.OX_VERBOSE = False

  # Use dynamic input
_C.INFERENCE.TRT_DYNAMIC = True

_C.INFERENCE.TRT_MIN_SHAPE = (1,3,256,256)

_C.INFERENCE.TRT_OPT_SHAPE = (1,3,700,700)

_C.INFERENCE.TRT_MAX_SHAPE = (1,3,1200,1200)

  # Use mix-precision (FP16)
_C.INFERENCE.TRT_AMP = True

  # Workspace size for export engine process (in GB), default=5 GB
_C.INFERENCE.TRT_WORKSPACE = 5