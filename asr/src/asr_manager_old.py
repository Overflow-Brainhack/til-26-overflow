import warnings
import torch
import io
import soundfile as sf
import numpy as np
from nemo.collections.asr.models import ASRModel
from numba.core.errors import NumbaWarning
from omegaconf import DictConfig  

warnings.filterwarnings("ignore", category=NumbaWarning)

class ASRManager:
    def __init__(self):
        """Initializes Parakeet-TDT for maximum T4 throughput."""
        # Load model directly to GPU and cast to FP16
        #model_path=r"./epoch10.nemo"
        model_path=r"./finetuned-parakeet-ctc-0.6b.nemo" # do not use .half()
        self.model = ASRModel.restore_from(model_path, map_location="cuda")

        # Only for CTC
        self.model.change_decoding_strategy(
            DictConfig({
                "strategy": "greedy_batch", # <--- Updated
                "compute_timestamps": False, 
                "use_batch_format": True
            })
        )
        self.model.cuda().eval()
        
        # Performance tuning
        self.model.decoding.use_cuda_graphs = False 

    @torch.inference_mode()
    def asr(self, audio_bytes: bytes) -> str:
        """High-speed in-memory transcription."""
        if not audio_bytes:
            return ""

        # 1. Faster Decoding: soundfile is significantly faster than librosa
        # We assume 16kHz input. If not, resample before sending to this function.
        with io.BytesIO(audio_bytes) as audio_stream:
            audio_signal, _ = sf.read(audio_stream, dtype='float32')

        # 2. Efficient Tensor Creation
        # Use pin_memory if coming from CPU, but here from_numpy + cuda() is optimal
        audio_tensor = torch.from_numpy(audio_signal).to(self.model.device, non_blocking=True)

        # 3. Optimized Transcription
        # Set batch_size=1 and disable verbose logging for speed
        # predictions = self.model.transcribe(
        #     [audio_tensor], 
        #     batch_size=1, 
        #     verbose=False,
        #     num_workers=0 # Faster for single-item inference
        # )

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            predictions = self.model.transcribe(
                [audio_tensor], 
                batch_size=1, 
                verbose=False,
                num_workers=0 
            )
            
        if not predictions:
            return ""

        result = predictions[0]
        return result.text if hasattr(result, 'text') else str(result)



# TODO
# Retrain parakeet-tdt-0.6b-v2 (not sure if v2 or v1)
# Train ctc-0.6b
# Train canary