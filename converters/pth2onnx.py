# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contain code for converting trained model into ONNX format
Refer from TensorRT example: tensorrt/bin/python/onnx_packnet
_____________________________________________________________________________
"""
# from icecream import ic
import sys
import os
from pathlib import Path

import torch
import onnx
import numpy as np
import argparse
import onnx_graphsurgeon as gs

from utils import experiment_loader, initial_logger, copyStateDict, get_cfg_defaults

sys.path.append("../")
from craft import CRAFT

logger = initial_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_onnx(args):
    """Load the akaocr network and export it to ONNX
    """
    logger.info("Converting detec pth to onnx...")

    model_path, model_config = experiment_loader(model_format='pth', data_path=args.weight)
    
    # Load config come with trained model
    cfg_detec = get_cfg_defaults()
    cfg_detec.merge_from_file(model_config)

    # Set output path for onnx files
    output_path = Path('../model_repository/detec_onnx/1')

    # Set name for onnx files
    output_detec = os.path.join(output_path, "detec_onnx.onnx")
    
    # Dummy input data for models
    input_tensor_detec = torch.randn((1, 3, 768, 768), requires_grad=False)
    input_tensor_detec=input_tensor_detec.cuda()
    input_tensor_detec=input_tensor_detec.to(device=device)

    # Load net
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path)))
    net = net.cuda()
    net.eval()

    # Convert the model into ONNX
    torch.onnx.export(net, input_tensor_detec, output_detec,
                      verbose=cfg_detec.INFERENCE.OX_VERBOSE, opset_version=cfg_detec.INFERENCE.OX_OPSET,
                      do_constant_folding=cfg_detec.INFERENCE.OX_DO_CONSTANT_FOLDING,
                      export_params=cfg_detec.INFERENCE.OX_EXPORT_PARAMS,
                      input_names=["input"],
                      output_names=["output", "output1"],
                      dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}})

    logger.info("Convert detec pth to ONNX sucess")

def main():
    parser = argparse.ArgumentParser(description="Exports akaOCR model to ONNX, and post-processes it to insert TensorRT plugins")
    parser.add_argument("--weight", required=False, help="Path to input model folder", default='../weights')
    
    args=parser.parse_args()

    build_onnx(args)

if __name__ == '__main__':
    main()
