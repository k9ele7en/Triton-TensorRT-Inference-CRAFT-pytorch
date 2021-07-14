#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains converter from torch .pth model to torchscript (jit) .pt model for Triton
_____________________________________________________________________________
"""
# from icecream import ic
import sys
import torch
import cv2
import numpy as np
import shutil
import os
import re
from PIL import Image
import argparse
from pathlib import Path
import torch.nn.functional as F
from utils import experiment_loader, initial_logger, copyStateDict

sys.path.append("../")
from craft import CRAFT

logger = initial_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_detec(args):
    model_path, _ = experiment_loader(model_format='pth', data_path=args.weight)

    logger.info(f"Converting to JIT (.pt) model from : {model_path}")
    
    # load net
    net = CRAFT()

    net.load_state_dict(copyStateDict(torch.load(model_path)))
    
    net = net.cuda()

    net.eval()

    # Prepare output name
    # Set output path for .pt file
    output_path = Path('../model_repository/detec_pt/1')

    # Set name for .pt files, default='<model_name>_pt.pt' as default of Triton config
    output_name = os.path.join(output_path, 'detec_pt.pt') 

    # An example input you would normally provide to your model's forward() method.
    x = torch.randn(1, 3, 768, 768).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net, x)

    # Save the TorchScript model
    traced_script_module.save(output_name)
    logger.info("Convert model detec from .pth to .pt (JIT) completed!")

def main():
    parser = argparse.ArgumentParser(description="Convert torch model (.pth) into torchscript model (.pt)")
    parser.add_argument("--weight", required=False, help="Path to input model folder", default='../weights')

    args=parser.parse_args()

    convert_detec(args)

if __name__ == '__main__':
    main()
