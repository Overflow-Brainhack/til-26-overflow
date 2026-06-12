import warnings
import argparse
import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils import model_utils
from numba.core.errors import NumbaWarning
from omegaconf import OmegaConf

# Filter specific jit warnings
warnings.filterwarnings("ignore", category=NumbaWarning)

# Optimize for Tensor Cores
torch.set_float32_matmul_precision("medium")


def train(model_name: str, config_path: str, save_path: str):
    """
    Generalized finetuning entry point for NeMo ASR models.
    """
    # 1. Load Pretrained Model (Automatically detects CTC, TDT, or RNNT)
    print(f"Loading pretrained model: {model_name}")
    model = ASRModel.from_pretrained(model_name)

    # 2. Load and Convert Config
    config = OmegaConf.load(config_path)
    cfg = model_utils.convert_model_config_to_dict_config(config)

    # 3. Initialize Trainer
    # pl.Trainer handles multi-GPU (DDP), precision (16-mixed), and logging
    trainer = pl.Trainer(**cfg.trainer)

    # 4. Bind Experiment Manager
    # Controls logging (W&B/TensorBoard), Checkpointing, and Early Stopping
    exp_manager(trainer, cfg.exp_manager)
    model.set_trainer(trainer)

    # 5. Setup Data Pipelines
    # Expects manifest-based data (path, duration, text)
    model.setup_training_data(train_data_config=cfg.model.train_ds)
    model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # 6. Override Optimization
    # Critical: This resets the LR schedule and optimizer from the checkpoint's
    # original state to your finetuning settings (e.g., lower LR for finetuning).
    model.setup_optimization(optim_config=cfg.optim)

    # 7. Execute Fine-Tuning
    trainer.fit(model)

    # 8. Export .nemo Artifact
    model.save_to(save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeMo ASR Finetuning")
    # nvidia/parakeet-ctc-0.6b
    # nvidia/parakeet-tdt-0.6b-v2
    # nvidia/parakeet-tdt-0.6b-v3
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained model name or path to .nemo file",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="finetuned_model.nemo",
        help="Filename for the exported model",
    )

    args = parser.parse_args()

    train(model_name=args.model, config_path=args.config, save_path=args.out)

