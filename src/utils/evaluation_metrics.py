import time
import torch
import numpy as np

def calculate_iou(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcule l'Intersection over Union (IoU) binaire.
    """
    # Binarisation des prédictions (les logits passent par une sigmoïde)
    preds_probs = torch.sigmoid(predictions)
    preds_bin = (preds_probs > threshold).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets_bin.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

class PerformanceProfiler:
    """
    Gestionnaire de contexte pour mesurer la latence (FPS) et la consommation VRAM.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.start_time = 0
        self.end_time = 0
        self.is_cuda = (device == "cuda" and torch.cuda.is_available())

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            torch.cuda.synchronize()
        self.end_time = time.time()

    def get_latency_ms(self):
        return (self.end_time - self.start_time) * 1000

    def get_fps(self, batch_size=1):
        latency_sec = self.end_time - self.start_time
        return batch_size / latency_sec if latency_sec > 0 else 0

    def get_peak_vram_mb(self):
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0
