# Check 
import onnx_asr
# import json
# data = {
#     "model_type": "nemo-conformer-tdt",
#     "features_size": 128,
#     "subsampling_factor": 8,
#  }

# with open('/workspace/finetuned-nemo-parakeet-tdt-0.6b-v2/config.json', 'w') as f:
#     json.dump(data, f, indent=4)


# This handles the loading of the multiple ONNX sub-files automatically
model_path = "/workspace/finetuned-nemo-parakeet-tdt-0.6b-v2"
model = onnx_asr.load_model(model='nemo-parakeet-tdt-0.6b-v2', path=model_path)
transcription = model.recognize("asr/sample_0.wav")
print(transcription)