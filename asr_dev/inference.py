import os
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/workspace/torch_compile_cache"
import warnings
import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from numba.core.errors import NumbaWarning
from omegaconf import open_dict, OmegaConf
import os
import random
import json
import time

random.seed(42)

warnings.filterwarnings("ignore", category=NumbaWarning)

torch.set_float32_matmul_precision("medium")
    
def predict(model, audio_path='asr/sample_0.wav'):
    with torch.inference_mode():
        predictions = model.transcribe(audio_path, verbose=False)[0]
        return predictions.text    

def inference(num_samples=100):
    tdt_path = "/workspace/runs/parakeet_tdt_0.6b_v2_finetuning/2026-05-10_09-10-39/checkpoints/parakeet_tdt_0.6b_v2_finetuning.nemo"
    ctc_path = r"/workspace/runs/parakeet_ctc_0.6b_finetuning/2026-05-10_08-20-27/checkpoints/parakeet_ctc_0.6b_finetuning.nemo"
    model = ASRModel.restore_from(tdt_path)
    model.eval()
    model.to('cuda')
    model.decoding.use_cuda_graphs = False
    #model.encoder = torch.compile(model.encoder, mode="default", dynamic=True)

    start_time = time.time()

    with open("asr/asr.jsonl", "r") as reader:
        data = [json.loads(line) for line in reader]
    
    data = random.sample(data, num_samples) 
    for sample in data:
        audio_path = os.path.join('asr', sample["audio"])
        transcription = predict(model, audio_path)
        print(f"Audio: {audio_path}\n Prediction: {transcription}\n Ground Truth: {sample['transcript']}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Inference of {num_samples} samples took {time_taken}s")
    return time_taken


if __name__ == '__main__':

    inference()

    # experiments tdt
    # with compile(default, dynamic) 124s
    # without compile 6s

    # 100 examples
    # tdt (without compile)
    # no .half() --> 12.6s
    # with .half() --> 77s


