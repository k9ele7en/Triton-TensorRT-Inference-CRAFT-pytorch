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
$ pip install -r requirements.txt
```
### 2. Install required environment for inference using Triton server
Check [./README_Triton.md](./README_Triton.md) for details. Install tools/packages included:
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

### 5. Model preparation before run Triton server:
a. Triton Inference Server inference: see details at [./README_Triton.md](./README_Triton.md)<br>
Initially, you need to run a (.sh) script to prepare Model Repo, then, you just need to run Docker image when inferencing.  Script get things ready for Triton server, steps covered:
- Convert downloaded pretrain into mutiple formats
- Locate converted model formats into Triton's Model Repository
- Run (Pull first if not exist) Triton Server image from NGC

Check if Server running correctly:
```
$ curl -v localhost:8000/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

Now everythings ready, start inference by:
- Run docker image of Triton server (replace mount -v path to your full path to model_repository):
```
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/triton-server-CRAFT-pytorch/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models
...
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| detec_onnx | 1       | READY  |
| detec_pt   | 1       | READY  |
| detec_trt  | 1       | READY  |
+------------+---------+--------+
I0714 00:37:55.265177 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0714 00:37:55.269588 1 http_server.cc:2887] Started HTTPService at 0.0.0.0:8000
I0714 00:37:55.312507 1 http_server.cc:2906] Started Metrics Service at 0.0.0.0:8002
```
Run infer by cmd: 
```
$ python infer_triton.py -m='detec_trt' -x=1 --test_folder='./images' -i='grpc' -u='localhost:8001'
Request 1, batch size 1s/sample.jpg
elapsed time : 0.9521937370300293s
```
#### Performance benchmarks: single image (sample.jpg), time in seconds
- Triton server: (gRPC-HTTP): <br>

    | Model format| gRPC (s)| HTTP (s) |
    |-------------|---------|----------|
    | TensoRT     | 0.946   | 0.952    |
    | Torchscript | 1.244   | 1.098    |
    | ONNX        | 1.052   | 1.060    |

- Classic Pytorch: 1.319s

#### Arguments
* `-m`: name of model with format
* `-x`: version of model
* `--test_folder`: input image/folder
* `-i`: protocol (HTTP/gRPC)
* `-u`: URL of corresponding protocol (HTTP-8000, gRPC-8001)
* ... (Details in ./infer_triton.py)

#### Notes:
- Error below is caused by wrong dynamic input shapes, check if the input image shape is valid to dynamic shapes in config.
```
inference failed: [StatusCode.INTERNAL] request specifies invalid shape for input 'input' for detec_trt_0_gpu0. Error details: model expected the shape of dimension 2 to be between 256 and 1200 but received 1216
```
b. Classic Pytorch (.pth) inference:
```
$ python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

The result image and socre maps will be saved to `./result` by default.
