# from nemo.export import TensorRTLLM  # For LLMs/Transducers
# from nemo.export.onnx_to_tensorrt import ONNXtoTensorRT

from nemo.collections.asr.models import ASRModel
import os

# 1. Load the model
path = "/workspace/runs/parakeet_tdt_0.6b_v2_finetuning/2026-05-10_09-10-39/checkpoints/parakeet_tdt_0.6b_v2_finetuning.nemo"
model = ASRModel.restore_from(path)
model.eval()

# 2. Step 1: Export to ONNX (Handles the TDT subnets automatically)
# This will likely create multiple files in the directory
export_dir = "/workspace/export_output"
os.makedirs(export_dir, exist_ok=True)

print("Exporting to ONNX...")
model.export(
    output=os.path.join(export_dir, "parakeet_v2.onnx"),
    onnx_opset_version=17,  # Safe for 5070 Ti
    verbose=False,
)

# 3. Step 2: Convert to TensorRT using 'trtexec'
# This is the most robust way if the Python module is missing.
# Run this in your terminal (already available in NVIDIA PyTorch/NeMo containers)
"""
trtexec --onnx=/workspace/export_output/parakeet_v2_encoder.onnx \
        --saveEngine=/workspace/export_output/encoder.plan \
        --fp16 \
        --minShapes=audio_signal:1x80x100 \
        --optShapes=audio_signal:16x80x1000 \
        --maxShapes=audio_signal:32x80x3000
"""

