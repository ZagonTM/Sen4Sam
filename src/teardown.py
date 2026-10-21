"""
Phase A helpers: instrument the frozen SAM 2.

Small, reusable measurements used by notebooks/01_base_model_teardown.ipynb:

  * parameter counts per block (to see where the model's "size" actually lives),
  * multiply-accumulate (MAC) counts per forward pass, via forward hooks,
  * capture of intermediate feature maps, via forward hooks,
  * effective receptive field (ERF), via the gradient of one output pixel w.r.t. the input,
  * latency and memory measurement.

Everything here is read-only instrumentation: it never changes the model.
"""

import time

import numpy as np
import torch


# --- Parameters ----------------------------------------------------------------

def params_of(module):
    """Number of trainable parameters in one module."""
    return sum(p.numel() for p in module.parameters())


def children_table(module):
    """Parameter count and share of each direct child of `module`."""
    total = params_of(module) or 1
    return [{"block": name, "type": type(child).__name__,
             "params_M": params_of(child) / 1e6,
             "share_pct": params_of(child) / total * 100}
            for name, child in module.named_children()]


# --- Compute cost (MACs) -------------------------------------------------------

def count_macs(root, *args, run=None):
    """Multiply-accumulate count for one forward pass.

    Counts Conv2d and Linear layers inside `root`: for a convolution each output
    element costs in_channels * kernel_h * kernel_w / groups multiply-adds; for a
    linear layer each output element costs in_features. Attention matmuls are not
    counted, so treat the number as a lower bound for transformer blocks.

    The pass executed is `run()` when given (to measure a higher-level entry point
    such as the image predictor's decoder, while counting `root`), else `root(*args)`.
    """
    total = [0]
    handles = []

    def conv_hook(layer, inputs, output):
        kh, kw = layer.kernel_size
        total[0] += output.numel() * layer.in_channels * kh * kw // layer.groups

    def linear_hook(layer, inputs, output):
        total[0] += output.numel() * layer.in_features

    for layer in root.modules():
        if isinstance(layer, torch.nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, torch.nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            run() if run is not None else root(*args)
    finally:
        for handle in handles:
            handle.remove()
    return total[0]


# --- Intermediate features -----------------------------------------------------

def capture(module, *args, targets, run=None):
    """Run a forward pass and return {name: raw output} for the given {name: module} hooks.

    The raw output is kept as-is (a tensor, or a list of tensors for multi-scale
    modules), so the caller decides how to unwrap it. As in `count_macs`, `run()`
    lets a higher-level entry point drive the pass while we watch `targets`.
    """
    store = {}

    def hook(name):
        def _store(hooked, inputs, output):
            store[name] = output
        return _store

    handles = [target.register_forward_hook(hook(name))
               for name, target in targets.items()]
    try:
        with torch.no_grad():
            run() if run is not None else module(*args)
    finally:
        for handle in handles:
            handle.remove()
    return store


# --- Effective receptive field -------------------------------------------------

def effective_receptive_field(encoder, size=1024, channel=0):
    """|d(output pixel) / d(input)| summed over colour channels.

    A large, spread-out gradient means the output pixel "sees" a large part of the
    image: that is its effective receptive field. We return a square numpy array.
    """
    image = torch.zeros(1, 3, size, size, device=next(encoder.parameters()).device,
                        requires_grad=True)
    features = encoder(image)["vision_features"]
    _, _, height, width = features.shape
    features[0, channel, height // 2, width // 2].backward()
    return image.grad.abs().sum(dim=1)[0].detach().cpu().numpy()


def erf_extent(erf, share=0.1):
    """Side length (in pixels) of the region holding `share` of the ERF mass."""
    total = erf.sum()
    if total <= 0:
        return 0
    flat = np.sort(erf.ravel())[::-1]
    reached = np.searchsorted(np.cumsum(flat), share * total) + 1
    return int(round(np.sqrt(reached)))


def pca_rgb(feature):
    """First three principal components of a (C, H, W) feature map, as an RGB image.

    A feature map has hundreds of channels, so we cannot look at it directly. The top
    three principal components usually capture where the channel activations vary, so
    we map them to R/G/B and stretch each one. This is a standard way to "look inside".
    """
    channels, height, width = feature.shape
    flat = feature.reshape(channels, -1).astype(np.float64)
    flat -= flat.mean(axis=1, keepdims=True)
    # eigh on the C x C Gram matrix is much cheaper than an SVD of C x pixels.
    # (errstate: Apple's Accelerate BLAS emits spurious FP warnings on matmul.)
    with np.errstate(all="ignore"):
        _, vectors = np.linalg.eigh(flat @ flat.T / flat.shape[1])
    components = vectors[:, -3:][:, ::-1]
    projection = (components.T @ flat).reshape(3, height, width)
    image = np.zeros((height, width, 3), np.float32)
    for i in range(3):
        low, high = np.percentile(projection[i], (2, 98))
        image[..., i] = np.clip((projection[i] - low) / (high - low + 1e-9), 0, 1)
    return (image * 255).astype(np.uint8)


# --- Latency and memory --------------------------------------------------------

def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_forward(module, *args, repeats=5, warmup=1):
    """Average wall-clock seconds per forward pass of `module(*args)`."""
    device = str(next(module.parameters()).device).split(":")[0]
    for _ in range(warmup):
        with torch.no_grad():
            module(*args)
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            module(*args)
    _sync(device)
    return (time.perf_counter() - started) / repeats


def time_call(fn, repeats=5, warmup=2, device=None):
    """Average wall-clock seconds per call of a zero-argument callable."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    for _ in range(warmup):
        fn()
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - started) / repeats


def allocated_memory_mb():
    """Currently allocated accelerator memory in MiB (MPS or CUDA; 0 on CPU)."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 2**20
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2**20
    return 0.0


def driver_memory_mb():
    """Total accelerator memory reserved from the driver in MiB (MPS or CUDA)."""
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 2**20
    return 0.0
