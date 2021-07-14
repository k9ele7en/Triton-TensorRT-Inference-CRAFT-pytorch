# Make sure you downloaded one of pretrain model and locate at /weight folder, link: https://github.com/clovaai/CRAFT-pytorch#test-instruction-using-pretrained-model
export failed=0 # To check if cmd run completed
export CUDA_VISIBLE_DEVICES=0 # Set specific GPU for TensorRT engine optimized for, start from 0

# Create model repository for Triton
mkdir -p ./model_repository/detec_pt/1 ./model_repository/detec_onnx/1 ./model_repository/detec_trt/1

# I. Convert pth model into (Torch JIT, ONNX, TensorRT)
cd converters/
python pth2jit.py || export failed=1
python pth2onnx.py || export failed=1
python onnx2trt.py || export failed=1
cd ..
# Convert ONNX to TensorRT can be done by trtexec command from TensorRT:
# /usr/src/tensorrt/bin/trtexec --onnx=./model_repository/detec_onnx/1/detec_onnx.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=./model_repository/detec_trt/1/detec_trt.plan


if [ ${failed} -ne 0 ]; then
        echo "Prepare Model Repo failed, check error on the terminal history above..."
      else
        echo "Convert source model into target formats and copy into Triton's Model Repository successfully."
        echo "Ready to run Triton inference server."
      fi

# III. Start Triton server image in container, mount Model Repo prepared into container volume
# Update the full path to data/model_repository follow deploy server path: "-v <full_path_to>/model_repository:/models"
sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/triton-server-CRAFT-pytorch/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models