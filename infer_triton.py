#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains main inference pipeline to Triton
_____________________________________________________________________________
"""
from icecream import ic
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from collections import OrderedDict

import triton_utils as triton

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Triton inference pipeline for CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1100, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='images/', type=str, help='folder path to input images')

# triton server
parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
parser.add_argument('-a',
                    '--async',
                    dest="async_set",
                    action="store_true",
                    required=False,
                    default=False,
                    help='Use asynchronous inference API')
parser.add_argument('--streaming',
                    action="store_true",
                    required=False,
                    default=False,
                    help='Use streaming inference API. ' +
                    'The flag is only available with gRPC protocol.')
parser.add_argument('-m',
                    '--model-name',
                    type=str,
                    required=False,
                    default='detec_trt',
                    help='Name of model in Triton Model Repo')
parser.add_argument('-x',
                    '--model-version',
                    type=str,
                    required=False,
                    default="1",
                    help='Version of model. Default is to use latest version.')
parser.add_argument('-b',
                    '--batch-size',
                    type=int,
                    required=False,
                    default=1,
                    help='Batch size. Default is 1.')
parser.add_argument('-u',
                    '--url',
                    type=str,
                    required=False,
                    default='localhost:8000',
                    help='Inference server URL. Default is localhost:8000.')
parser.add_argument('-i',
                    '--protocol',
                    type=str,
                    required=False,
                    default='HTTP',
                    help='Protocol (HTTP/gRPC) used to communicate with ' +
                    'the inference service. Default is HTTP.')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(args, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    # x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # if cuda:
    #     x = x.cuda()

    # send request to triton server
    y = triton.triton_requester(args, x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()


    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    t = time.time()
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(args, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

        # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask_triton.jpg'
        # cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, method='triton')

    print("elapsed time : {}s".format(time.time() - t))

# Example cmd:
# python infer_triton.py -m='detec_trt' -x=1 --test_folder='./images' -i='grpc' -u='localhost:8001'
# python infer_triton.py -m='detec_onnx' -x=1 --test_folder='./images'
# python infer_triton.py -m='detec_pt' -x=1 --test_folder='./images'