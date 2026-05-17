import os
import torch
import soundfile as sf
import io
import warnings
from numba.core.errors import NumbaWarning
from nemo.collections.asr.models import EncDecCTCModelBPE
from omegaconf import OmegaConf  # Needed to change decoding strategy
import numpy as np

warnings.filterwarnings("ignore", category=NumbaWarning)

class ASRManager:
    def __init__(self):
        self.model = EncDecCTCModelBPE.restore_from(
            "./finetuned-parakeet-ctc-0.6b.nemo"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # FIX 1: Switch decoding strategy to 'greedy_batch' for a free speedup
        cfg = OmegaConf.create({"strategy": "greedy_batch"})
        self.model.change_decoding_strategy(cfg)

    def asr(self, audio_bytes: bytes) -> str:
        # FIX 2: Force soundfile to read directly into float32 
        with io.BytesIO(audio_bytes) as f:
            audio_signal, _ = sf.read(f, dtype='float32')
      
        with torch.inference_mode():
            result = self.model.transcribe([audio_signal], verbose=False)
            
        if not result:
            return ""
            
        return result[0].text if hasattr(result[0], 'text') else result[0]