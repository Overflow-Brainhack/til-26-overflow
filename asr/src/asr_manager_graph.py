import torch
import soundfile as sf
import io
import numpy as np
import warnings
from numba.core.errors import NumbaWarning
from nemo.collections.asr.models import EncDecRNNTBPEModel

warnings.filterwarnings("ignore", category=NumbaWarning)

class ASRManager:
    def __init__(self):
        # Fixed duration in samples (e.g., 5 seconds at 16kHz = 80,000)
        self.fixed_length = 50* 16000 
        
        self.model = EncDecRNNTBPEModel.restore_from("./finetuned-parakeet-tdt-0.6b-v2.nemo")
        self.device = torch.device("cuda")
        self.model.to(self.device).eval()

        # Enable CUDA Graphs in the NeMo decoding config
        self.model.decoding.use_cuda_graphs = True
        
        # Warmup: CUDA Graphs capture the stream during the first fixed-length forward pass
        self._warmup()

    def _warmup(self):
        """Initial pass to trigger CUDA Graph capture inside NeMo's transcribe."""
        dummy_signal = np.zeros(self.fixed_length, dtype=np.float32)
        print(dummy_signal.size())
        with torch.inference_mode():
            self.model.transcribe([dummy_signal], verbose=False)

    def asr(self, audio_bytes: bytes) -> str:
        with io.BytesIO(audio_bytes) as f:
            audio_signal, _ = sf.read(f)

        # Fast NumPy padding/truncating to fixed bucket
        sig_len = len(audio_signal)
        if sig_len < self.fixed_length:
            audio_signal = np.pad(audio_signal, (0, self.fixed_length - sig_len))

        with torch.inference_mode():
            # transcribe() will now use the captured graph due to static shape
            result = self.model.transcribe([audio_signal], verbose=False)
            
        return result[0].text if result else ""