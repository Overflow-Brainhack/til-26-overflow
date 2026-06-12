"""Manages the ASR model."""

import torch
import soundfile as sf
import io
import warnings
from numba.core.errors import NumbaWarning
from nemo.collections.asr.models import EncDecRNNTBPEModel

warnings.filterwarnings("ignore", category=NumbaWarning)


class ASRManager:
    def __init__(self):
        self.fixed_length = 50 * 16000
        self.model = EncDecRNNTBPEModel.restore_from(
            "./finetuned-parakeet-tdt-0.6b-v2.nemo"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.model.decoding.use_cuda_graphs = False

    def asr(self, audio_bytes: bytes) -> str:
        with io.BytesIO(audio_bytes) as f:
            audio_signal, _ = sf.read(f, dtype="float32")

        with torch.inference_mode():
            result = self.model.transcribe([audio_signal], verbose=False)

        return result[0].text if result else ""
