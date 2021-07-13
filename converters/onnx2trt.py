#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains code for convert from onnx to tensorRT engine
_____________________________________________________________________________
"""
import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2
from pathlib import Path

import onnx
import argparse

from utils import experiment_loader, initial_logger, copyStateDict, get_cfg_defaults

from config import _C
logger = initial_logger()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a=(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def build_detec_engine(onnx_path, using_half, engine_file, dynamic_input=True, workspace_size=5, 
                min_shape=(1,3,256,256), opt_shape=(1,3,700,700), max_shape=(1,3,1200,1200)):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(int(workspace_size))
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        
        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("input", min_shape, opt_shape, max_shape) 
            config.add_optimization_profile(profile)
   
        return builder.build_engine(network, config) 


if __name__ == '__main__':
    # set dynamic input size in associate function, find: profile.set_shape
    parser = argparse.ArgumentParser(description="Convert ONNX model into TensorRT")
    parser.add_argument("--weight", required=False, help="Path to input model folder", default='../weight')

    args=parser.parse_args()
    logger.info("Converting detec ONNX to TensorRT engine...")
    

    _, model_config = experiment_loader(model_format='pth', data_path=args.weight)
    model_path = '../model_repository/detec_onnx/1/detec_onnx.onnx'
    # Process names
    # Set output path for onnx files
    output_path = Path('../model_repository/detec_trt/1')

    # Set name for onnx files
    output_detec = os.path.join(output_path, "detec_trt.plan")

    cfg_detec = get_cfg_defaults()
    cfg_detec.merge_from_file(model_config)

    # start build
    detec_engine=build_detec_engine(model_path, cfg_detec.INFERENCE.TRT_AMP, output_detec, dynamic_input=cfg_detec.INFERENCE.TRT_DYNAMIC, 
                workspace_size=cfg_detec.INFERENCE.TRT_WORKSPACE, min_shape=cfg_detec.INFERENCE.TRT_MIN_SHAPE, 
                opt_shape=cfg_detec.INFERENCE.TRT_OPT_SHAPE, max_shape=cfg_detec.INFERENCE.TRT_MAX_SHAPE)
    with open(output_detec, "wb") as f:
        f.write(detec_engine.serialize())
    logger.info('Build RT engine of detec successfully!')

# Sample run cmd: for ex, engine optimze for gpu:0
# CUDA_VISIBLE_DEVICES=0 python onnx2trt.py