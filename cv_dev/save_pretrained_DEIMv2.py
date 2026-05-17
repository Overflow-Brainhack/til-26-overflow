import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin

import sys

sys.path.append("/home/dev/DEIMv2/")

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        else:
            self.backbone = HGNetv2(**config["HGNetv2"])
        if "LiteEncoder" in config:
            self.encoder = LiteEncoder(**config["LiteEncoder"])
        else:
            self.encoder = HybridEncoder(**config["HybridEncoder"])
        if "DEIMTransformer" in config:
            self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        else:
            self.decoder = DFINETransformer(**config["DFINETransformer"])
        self.postprocessor = PostProcessor(**config["PostProcessor"])

    def forward(self, x, orig_target_sizes):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)

        return x


deimv2_l_config = {
    "DINOv3STAs": {
        "name": "dinov3_vits16",
        "embed_dim": 224,
        "interaction_indexes": [5, 8, 11],
        "num_heads": None,
        "conv_inplane": 32,
        "hidden_dim": 224,
    },
    "HybridEncoder": {
        "in_channels": [224, 224, 224],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 224,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 896,
        "dropout": 0.0,
        "enc_act": "gelu",
        "expansion": 1.0,
        "depth_mult": 1,
        "act": "silu",
        "version": "deim",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "DEIMTransformer": {
        "feat_channels": [224, 224, 224],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 224,
        "num_levels": 3,
        "num_layers": 4,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "activation": "silu",
        "mlp_act": "silu",
        "dim_feedforward": 1792,
        "eval_spatial_size": [640, 640],
    },
    "PostProcessor": {"num_top_queries": 300},
}

deimv2_l = DEIMv2(deimv2_l_config)
deimv2_l_hf = DEIMv2.from_pretrained("Intellindust/DEIMv2_DINOv3_L_COCO")
torch.save({"model": deimv2_l_hf.state_dict()}, "deimv2-hf.pth")
