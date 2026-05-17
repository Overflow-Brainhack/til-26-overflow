import tempfile
import torch
import os
from nemo.collections.asr.models import EncDecRNNTBPEModel
import warnings
from numba.core.errors import NumbaWarning

warnings.filterwarnings("ignore", category=NumbaWarning)

class ASRManager:
    def __init__(self):
        self.model = EncDecRNNTBPEModel.restore_from(
            "./finetuned-parakeet-tdt-0.6b-v2.nemo"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
        self.model.eval()

        self.model.decoding.use_cuda_graphs = False 

    def asr(self, audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        
        try:
            with torch.inference_mode():
                result = self.model.transcribe([path], verbose=False)
        finally:
            os.remove(path)
            
        return result[0].text if result else ""
        
