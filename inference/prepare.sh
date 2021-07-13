# Make sure you downloaded one of pretrain model and locate at /weight folder, link: https://github.com/clovaai/CRAFT-pytorch#test-instruction-using-pretrained-model
export weight='../weight'
export failed=0 # To check if cmd run completed
export CUDA_VISIBLE_DEVICES=0 # Set specific GPU for TensorRT engine optimized for, start from 0
# Create model repository for Triton
mkdir -p ../../data/model_repository/detec_pt/1 ../../data/model_repository/detec_onnx/1 ../../data/model_repository/detec_trt/1


# I. Convert pth model into (Torch JIT, ONNX, TensorRT)
python ../converters/pth2jit.py --weight=$weight || export failed=1
python ../converters/pth2onnx.py --weight=$weight || export failed=1
python ../converters/onnx2trt.py --weight=$weight || export failed=1
# /usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/detec_onnx.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/detec_trt.engine


# II. Copy models into Model Repo for Triton server
if [ ${failed} -ne 0 ]; then
        echo "Convert failed, check error on the terminal history above..."
      else
        # pt
        cp ../../data/exp_detec/test/detec_pt.pt ../../data/model_repository/detec_pt/1/detec_pt.pt || export failed=1
        
        # onnx
        cp ../../data/exp_detec/test/detec_onnx.onnx ../../data/model_repository/detec_onnx/1/detec_onnx.onnx || export failed=1

        # trt
        cp ../../data/exp_detec/test/detec_trt.engine ../../data/model_repository/detec_trt/1/detec_trt.plan || export failed=1
      fi


if [ ${failed} -ne 0 ]; then
        echo "Prepare Model Repo failed, check error on the terminal history above..."
      else
        echo "Convert source model into target formats and copy into Triton's Model Repository successfully."
        echo "Ready to run Triton inference server."
      fi

# III. Start Triton server image in container, mount Model Repo prepared into container volume
# Update the full path to data/model_repository follow deploy server path: "-v <full_path_to>/ocr-components-triton/akaocr/data/model_repository:/models"
sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/triton-server-CRAFT-pytorch/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models