<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# About
This is an introduction and basic guideline to run inference using single model method or NVIDIA Triton Inference Server

## NVIDIA Triton Inference Server
NVIDIA Triton Inference Server is an open-source inference serving software that simplifies inference serving for an organization by addressing the above complexities. Triton provides a single standardized inference platform which can support running inference on multi-framework models, on both CPU and GPU, and in different deployment environments such as datacenter, cloud, embedded devices, and virtualized environments.

It natively supports multiple framework backends like TensorFlow, PyTorch, ONNX Runtime, Python, and even custom backends. It supports different types of inference queries through advanced batching and scheduling algorithms, supports live model updates, and runs models on both CPUs and GPUs. Triton is also designed to increase inference performance by maximizing hardware utilization through concurrent model execution and dynamic batching. Concurrent execution allows you to run multiple copies of a model, and multiple different models, in parallel on the same GPU. Through dynamic batching, Triton can dynamically group together inference requests on the server-side to maximize performance.

## Open Neural Network Exchange (ONNX)
Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

## TensorRT
TensorRT is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution.
After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.

## PyCUDA
PyCUDA lets you access Nvidia's CUDA parallel computation API from Python.
For more information, check at: https://documen.tician.de/pycuda/

## NVIDIA Container Toolkit (NVIDIA-Docker)
The NVIDIA Container Toolkit allows users to build and run GPU accelerated Docker containers. The toolkit includes a container runtime library and utilities to automatically configure containers to leverage NVIDIA GPUs.

## How to run inference pipeline
There are multiple method to infer the detec pipeline, devided into 2 main types:
- Using single model format: supported formats are: Pytorch (.pth), ONNX (.onnx)
- Using NVIDA Triton Server: use Docker container as a server, with one Model Repository to refer to. Support Torch JIT (.pt), ONNX (.onnx), TensorRT engine (.plan)
* Next release: no need to install complicated environment components, build Docker image and run with repository sourcecode.
If you run the pipeline using single model format, run as below, no need environment installation as Triton server.
```
$ cd inference
Run infer_triton.py with target method (pth/onnx/tensorrt) and suitable arguments of that method. Use single format, for ex pth format:
$ python infer_triton.py -m='detec_pt' -x=1 --test_folder='./images'
```
## I. Setup environment and tools
First, update your PIP:
```
$ python3 -m pip install --upgrade setuptools pip 
```
1. ONNX: install ONNX packages by pip and conda packages
```
$ python3 -m pip install nvidia-pyindex
$ conda install -c conda-forge onnx
$ pip install onnx_graphsurgeon 
```
Note: If convert pth to onnx get error (libstdc++.so.6: version `GLIBCXX_3.4.22' not found), fix by run below commands:
```
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
$ sudo apt-get update 
$ sudo apt-get install gcc-4.9 
$ sudo apt-get install --only-upgrade libstdc++6 
```
2. TensorRT (for detail install instruction, check at: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#install)
- Login and download local repo file that matches the Ubuntu version and CPU architecture that you are using. from https://developer.nvidia.com/tensorrt
- Install downloaded deb package as guide (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian): <br>
    - Install TensorRT from the Debian local repo package. Replace ubuntuxx04, cudax.x, trt8.x.x.x-ea and yyyymmdd with your specific OS version, CUDA version, TensorRT version and package date.
    ```
    $ os="ubuntuxx04"
    $ tag="cudax.x-trt8.x.x.x-ea-yyyymmdd"
    $ sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
    $ sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

    $ sudo apt-get update
    $ sudo apt-get install tensorrt
    ```
    - Note this error when running above cmd, for ex. install TensorRT v7.2.3.4:
    ```
    $ sudo apt-get install tensorrt
    The following packages have unmet dependencies:
    tensorrt : Depends: libnvinfer-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvinfer-plugin-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvparsers-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvonnxparsers-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvinfer-samples (= 7.2.3-1+cuda11.1) but it is not going to be installed
    E: Unable to correct problems, you have held broken packages.
    ```
    - *Reason: APT-GET choose wrong version of dependencies to install as required by target TensorRT version. Run the followings cmd to solve: sudo apt-get -y install <dependency_name>=<target_version>...*
    ```
    $ sudo apt-get -y install libnvinfer-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-plugin-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvparsers-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvonnxparsers-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-samples=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-plugin-dev=7.2.3-1+cuda11.1
    ...
    Now try again:
    $ sudo apt-get -y install tensorrt
    ```
    - If using Python 3.x:
    ```
    $ sudo apt-get install python3-libnvinfer-dev
    ```
    The following additional packages will be installed:
    python3-libnvinfer
    If you would like to run the samples that require ONNX graphsurgeon or use the Python module for your own project, run:
    ```
    $ sudo apt-get install onnx-graphsurgeon
    ```
    Verify the installation.
    ```
    $ dpkg -l | grep TensorRT
    ```
- Install pip packages (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
    - If your pip and setuptools Python modules are not up-to-date, then use the following command to upgrade these Python modules. If these Python modules are out-of-date then the commands which follow later in this section may fail.
    ```
    $ python3 -m pip install --upgrade setuptools pip
    ```
    You should now be able to install the nvidia-pyindex module.
    ```
    $ python3 -m pip install nvidia-pyindex
    ```
    - Install the TensorRT Python wheel.
    ```
    $ python3 -m pip install --upgrade nvidia-tensorrt
    ```
    The above pip command will pull in all the required CUDA libraries and cuDNN in Python wheel format because they are dependencies of the TensorRT Python wheel. Also, it will upgrade nvidia-tensorrt to the latest version if you had a previous version installed.

    - To verify that your installation is working, use the following Python commands to:
    ```
    python3
    >>> import tensorrt
    >>> print(tensorrt.__version__)
    >>> assert tensorrt.Builder(tensorrt.Logger())
    ```

3. PyCUDA (for details, check at: https://wiki.tiker.net/PyCuda/Installation/Linux/#step-1-download-and-unpack-pycuda)
-  Step 1: Download source of pip package tar.gz and unpack PyCUDA from https://pypi.org/project/pycuda/#files
```
$ tar xfz pycuda-VERSION.tar.gz
```
- Step 2: Install Numpy
PyCUDA is designed to work in conjunction with numpy, Python's array package. Here's an easy way to install it, if you do not have it already:
```
$ cd pycuda-VERSION
$ su -c "python distribute_setup.py" # this will install distribute
$ su -c "easy_install numpy" # this will install numpy using distribute
```
- Step 3: Build PyCUDA
Next, just type:
```
Install make if needed:
$ sudo apt-get install -y make

Start building:
$ cd pycuda-VERSION # if you're not there already
$ python configure.py --cuda-root=/where/ever/you/installed/cuda
$ su -c "make install"
```
- Step 4: Test PyCUDA
If you'd like to be extra-careful, you can run PyCUDA's unit tests:
```
$ cd pycuda-VERSION/test
$ python test_driver.py
```
4. NVIDIA Container Toolkit
- Setting up Docker (Follow the official instructions for more details and post-install actions at: https://docs.docker.com/engine/install/ubuntu/)
Docker-CE on Ubuntu can be setup using Docker’s official convenience script:
```
$ curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```
- Setting up NVIDIA Container Toolkit
    - Setup the stable repository and the GPG key:
    ```
    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```
    - Install the nvidia-docker2 package (and dependencies) after updating the package listing:
    ```
    $ sudo apt-get update
    $ sudo apt-get install -y nvidia-docker2
    ```
    - Restart the Docker daemon to complete the installation after setting the default runtime:
    ```
    $ sudo systemctl restart docker
    ```
    - At this point, a working setup can be tested by running a base CUDA container:
    ```
    $ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```

## Note:
- TensorRT need to be installed with the same version as used in Triton server Docker image, so that the engine created by TensorRT can be loaded into Model Repo of Triton. For ex, in Triton server version 21.05, TensorRT's version is 7.2.3.4
- Trained model which saved by torch.save (usually .pth) must be convert into torchscript by torch.jit.save (into model.pt as default name of Triton).
## II. Prepare Triton server docker image, source repo or client docker image
Pull repo, image, and prepare models (Where <xx.yy> is the version of Triton that you want to use):
```
$ sudo docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
$ git clone https://github.com/huukim911/Triton-TensorRT-Inference-CRAFT-pytorch.git
Run the .sh script to convert model into target formats, prepare Model Repo and start Triton server container:
$ cd Triton-TensorRT-Inference-CRAFT-pytorch
$ sh prepare.sh
Convert source model into target formats and copy into Triton's Model Repository successfully.
```
## III. Run the server and client to infer (included in .sh script):
Run server in container and client in cmd
```
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v <full_path_to/model_repository>:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models

For example, run on server with full path "/home/maverick911/repo/Triton-TensorRT-Inference-CRAFT-pytorch
/model_repository":
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/Triton-TensorRT-Inference-CRAFT-pytorch
/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models

+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| detec_pt             | 1       | READY  |
| detec_trt            | 1       | READY  |
....
I0611 04:10:23.026207 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0611 04:10:23.036976 1 http_server.cc:2987] Started HTTPService at 0.0.0.0:8000
I0611 04:10:23.080860 1 http_server.c9:2906] Started Metrics Service at 0.0.0.0:8002
```
2. Infer by client in cmd (this repo), with method (triton), model name (<model_type>_\<format>), version (not required). For ex:
```
$ cd Triton-TensorRT-Inference-CRAFT-pytorch/
$ python infer_triton.py -m='detec_trt' -x=1 --test_folder='./images'
Request 1, batch size 1s/sample.jpg
elapsed time : 0.9521937370300293s
```
```
$ python infer_triton.py -m='detec_pt' -x=1 --test_folder='./images' -i='grpc' -u='localhost:8001'
Request 1, batch size 1s/sample.jpg
elapsed time : 1.244419813156128s
```
-------
Run server in container and client sdk in container:
1. Start the server side:
```
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/Triton-TensorRT-Inference-CRAFT-pytorch/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models

+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| detec_pt             | 1       | READY  |
| detec_trt            | 1       | READY  |
....
I0611 04:10:23.026207 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0611 04:10:23.036976 1 http_server.cc:2987] Started HTTPService at 0.0.0.0:8000
I0611 04:10:23.080860 1 http_server.c9:2906] Started Metrics Service at 0.0.0.0:8002
```
2. Start client image to start inferencing (shell), mount client src into container:
```
$ sudo docker run -it --rm --net=host -v <full_path/to/repo>:/workspace/client nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```
3. Use infer_triton.py as example above to run.
