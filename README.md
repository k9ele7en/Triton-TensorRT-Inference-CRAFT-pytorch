## Advanced Triton Inference Pipeline for CRAFT (Character-Region Awareness For Text detection)
- High performance and advanced inference for text detection with Triton Inference Server. Model format included: TensorRT, ONNX, Torchscript. <br>
- CRAFT: (forked from https://github.com/clovaai/CRAFT-pytorch)
Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

### Sample Results

### Overview
Implementation new inference pipeline using NVIDIA Triton Inference Server for CRAFT text detector in Pytorch.

## Updates
**13 Jul, 2021**: Initial update, converters done.


## Getting started
### 1. Install dependencies
#### Requirements
```
pip install -r requirements.txt
```
### 2. Install required environment for inference using Triton server
Check inference/README.md for details. Install tools/packages included:
- TensorRT
- Docker
- nvidia-docker
- PyCUDA
...

### 3. Training
The code for training is not included in this repository, as ClovaAI provided.


### 4. Inference instruction using pretrained model
- Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

### 5. Run preparation script before run Triton server:
a. Triton Inference Server inference: see detail at inference/README.md <br>
Run the preparation script to get things ready for Triton server, steps covered:
- Convert downloaded pretrain into mutiple formats
- Locate converted model formats into Triton's Model Repository
- Run (Pull first if not exist) Triton Server image from NGC

Now Check Ready status for begin infer...
Example infer by TensorRT:
```
python 
```

#### Arguments
* `--trained_model`: pretrained model

b. Classic Pytorch (.pth) inference:
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

The result image and socre maps will be saved to `./result` by default.
