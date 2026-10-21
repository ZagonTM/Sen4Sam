"""
SAM 2.1 model setup for prompt-free (automatic) image segmentation.

This module only builds the model and the automatic mask generator.
"""

from pathlib import Path

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Repository root (the folder that contains src/, checkpoints/, configs/).
# Resolving paths from here keeps the script working no matter the working
# directory it is launched from (local terminal, VS Code, or Colab).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Model configuration: architecture name + pre-trained weights.
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT = str(REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt")


def get_device():
    """Pick the best available device.

    CUDA on Colab, MPS on Apple Silicon, otherwise CPU. CUDA is checked
    first so a machine that offers both uses the GPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(config=CONFIG, checkpoint=CHECKPOINT, device=None):
    """Load the pre-trained SAM 2 model and put it in eval mode."""
    model = build_sam2(config, checkpoint, device=device or get_device())
    model.eval()
    return model


def build_mask_generator(model, points_per_side=32, **kwargs):
    """Build the prompt-free mask generator.

    "Prompt-free" means that the generator places its own grid of points over the
    image (points_per_side per side) and returns one mask per point, filtered by
    predicted quality. Extra keyword arguments (pred_iou_thresh,
    stability_score_thresh, ...) are passed through to SAM 2.
    """
    return SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side, **kwargs)


if __name__ == "__main__":
    # Build the model and report the device it actually runs on.
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {model.device} | parameters: {n_params/1e6:.2f} M")
