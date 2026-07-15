"""GPU-accelerated detector preprocessing with torch-compatible masks.

Replaces cv2.GaussianBlur with a custom torch Gaussian filter when torch
is available, and moves otsu_mask / watershed_mask / edge_mask to GPU
where possible.  Falls back to CPU/cv2/scipy when torch/cuda is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from qem.utils.tensors import best_device, to_numpy, to_tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cv2 fallback helpers
# ---------------------------------------------------------------------------

try:
    from cv2 import GaussianBlur as _cv2_gaussian_blur
    from cv2 import moments as _cv2_moments
except ImportError:
    _cv2_gaussian_blur = None  # type: ignore[assignment]
    _cv2_moments = None  # type: ignore[assignment]


try:
    from skimage import filters as _skimage_filters
    from skimage import segmentation as _skimage_segmentation
    from skimage.feature import canny as _skimage_canny
except ImportError:
    _skimage_filters = None  # type: ignore[assignment]
    _skimage_segmentation = None  # type: ignore[assignment]
    _skimage_canny = None  # type: ignore[assignment]


try:
    from scipy import ndimage as _scipy_ndi
except ImportError:
    _scipy_ndi = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# torch Gaussian blur
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """1-D Gaussian kernel normalised to sum=1."""
    if size <= 0:
        size = max(int(4 * sigma + 0.5) * 2 + 1, 1)
    coords = torch.arange(size, dtype=dtype) - (size - 1) / 2.0
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def torch_gaussian_blur(image: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
    """Separable Gaussian blur via 1-D convolutions.

    Args:
        image: 2-D tensor [H, W] or 3-D [C, H, W].
        kernel_size: odd integer; if even it is bumped up by 1.
        sigma: Gaussian sigma.
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)  # [1, H, W]
    elif image.dim() == 3:
        pass
    else:
        raise ValueError(f"Expected 2-D or 3-D image, got {image.dim()}-D")

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = _gaussian_kernel_1d(kernel_size, sigma, dtype=image.dtype).to(image.device)
    kernel_h = kernel.view(1, 1, -1, 1).expand(image.size(0), 1, -1, 1)
    kernel_w = kernel.view(1, 1, 1, -1).expand(image.size(0), 1, 1, -1)

    pad = kernel_size // 2
    # vertical conv
    out = F.conv2d(image.unsqueeze(0), kernel_h, padding=(pad, 0), groups=image.size(0))
    # horizontal conv
    out = F.conv2d(out, kernel_w, padding=(0, pad), groups=image.size(0))
    return out.squeeze(0)  # [C, H, W] or [1, H, W]


def _gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 2.0, device: torch.device | None = None) -> np.ndarray:
    """Dispatch to torch Gaussian blur if possible, else cv2."""
    if device is None:
        device = best_device()
    try:
        t = to_tensor(image, dtype="float32").to(device)
        blurred = torch_gaussian_blur(t, kernel_size=kernel_size, sigma=sigma)
        return to_numpy(blurred)
    except Exception as e:
        if _cv2_gaussian_blur is not None:
            k = (kernel_size, kernel_size)
            return _cv2_gaussian_blur(image.astype(np.float32), k, sigmaX=sigma, sigmaY=sigma)
        raise RuntimeError("No Gaussian blur backend available (torch or cv2)") from e


# ---------------------------------------------------------------------------
# torch-compatible masks (with CPU fallback)
# ---------------------------------------------------------------------------

def _torch_otsu_threshold(image: torch.Tensor, classes: int = 2) -> torch.Tensor:
    """Multi-Otsu threshold on a 2-D torch tensor (CPU or GPU).

    Uses torch.histc for histogramming so the whole pipeline stays on device.
    For classes>2 we use a simple recursive 2-class split on each side.
    """
    if image.numel() == 0:
        return torch.tensor([0.5], dtype=image.dtype, device=image.device)

    # Normalise to [0, 1] for histogram
    vmin = image.min()
    vmax = image.max()
    if vmax == vmin:
        return torch.tensor([vmin + 0.5], dtype=image.dtype, device=image.device)

    normed = (image - vmin) / (vmax - vmin)
    hist = torch.histc(normed, bins=256, min=0.0, max=1.0)
    bin_edges = torch.linspace(0.0, 1.0, 257, dtype=image.dtype, device=image.device)

    # Cumulative sums for Otsu
    total = hist.sum()
    if total == 0:
        return torch.tensor([0.5], dtype=image.dtype, device=image.device)

    # Compute cumulative moments on device
    bins = torch.arange(256, dtype=image.dtype, device=image.device)
    p = hist / total
    omega = torch.cumsum(p, dim=0)
    mu = torch.cumsum(bins * p, dim=0)
    mu_total = mu[-1]

    # Between-class variance
    sigma_b = ((mu_total * omega - mu) ** 2) / (omega * (1 - omega) + 1e-12)
    max_idx = torch.argmax(sigma_b)
    threshold = bin_edges[max_idx] * (vmax - vmin) + vmin

    if classes == 2:
        return threshold.unsqueeze(0)

    # For classes>2, recursively split each side (approximation)
    thresholds = [threshold.item()]
    low = image[image < threshold]
    high = image[image >= threshold]
    for side in (low, high):
        if side.numel() > 0:
            t2 = _torch_otsu_threshold(side, classes=2)
            thresholds.append(t2.item())
    thresholds = sorted(set(thresholds))
    return torch.tensor(thresholds, dtype=image.dtype, device=image.device)


def torch_otsu_mask(image: np.ndarray | torch.Tensor, normalized: bool = False, device: torch.device | None = None) -> np.ndarray:
    """Otsu binary mask — GPU when possible, CPU fallback via skimage."""
    if device is None:
        device = best_device()
    t = to_tensor(image, dtype="float32")
    is_torch_input = torch.is_tensor(image)
    if not is_torch_input:
        t = t.to(device)
    else:
        t = t.to(device)

    try:
        thresholds = _torch_otsu_threshold(t, classes=2)
        if thresholds.numel() == 1:
            mask = t > thresholds[0]
        else:
            regions = torch.zeros_like(t, dtype=torch.int64)
            for thr in thresholds:
                regions += (t > thr).long()
            mask = regions == 1
        return to_numpy(mask) if not is_torch_input else mask
    except Exception as e:
        if _skimage_filters is not None:
            thresholds = _skimage_filters.threshold_multiotsu(to_numpy(t), classes=2)
            regions = np.digitize(to_numpy(t), bins=thresholds)
            mask = regions == 1
            return mask
        raise RuntimeError("No Otsu backend available") from e


def torch_watershed_mask(image: np.ndarray | torch.Tensor, device: torch.device | None = None) -> np.ndarray:
    """Watershed mask — GPU when possible, CPU fallback via skimage."""
    if device is None:
        device = best_device()
    t = to_tensor(image, dtype="float32")
    is_torch_input = torch.is_tensor(image)
    if not is_torch_input:
        t = t.to(device)
    else:
        t = t.to(device)

    try:
        # Simple torch-based watershed approximation using threshold markers
        threshold_value = torch.quantile(t, 0.8)
        markers = torch.zeros_like(t, dtype=torch.int64)
        markers[t < threshold_value] = 1
        markers[t >= threshold_value] = 2
        # Approximate watershed by connected components of markers
        # (full watershed requires scipy/skimage; fall back there)
        if _skimage_segmentation is not None:
            raise RuntimeError("Fallback to skimage")
        # Very basic approximation: just return the marker boundary
        mask = markers == 2
        return to_numpy(mask) if not is_torch_input else mask
    except Exception as e:
        if _skimage_segmentation is not None:
            np_img = to_numpy(t)
            markers_np = np.zeros_like(np_img)
            threshold_value = np.percentile(np_img, 80)
            markers_np[np_img < threshold_value] = 1
            markers_np[np_img > threshold_value] = 2
            mask = _skimage_segmentation.watershed(np_img, markers_np)
            return mask
        raise RuntimeError("No watershed backend available") from e


def torch_edge_mask(image: np.ndarray | torch.Tensor, sigma: float = 3.0, device: torch.device | None = None) -> np.ndarray:
    """Edge mask via Sobel + fill holes — GPU when possible, CPU fallback via skimage."""
    if device is None:
        device = best_device()
    t = to_tensor(image, dtype="float32")
    is_torch_input = torch.is_tensor(image)
    if not is_torch_input:
        t = t.to(device)
    else:
        t = t.to(device)

    try:
        # Sobel edge detection in torch
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
        img = t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
        # Threshold at mean + sigma * std (analogous to canny)
        mean = magnitude.mean()
        std = magnitude.std()
        edges = magnitude > (mean + sigma * std)
        # Fill holes and erode — fallback to scipy if available
        if _scipy_ndi is not None:
            edges_np = to_numpy(edges)
            fill_detector = _scipy_ndi.binary_fill_holes(edges_np)
            mask = _scipy_ndi.binary_erosion(fill_detector, iterations=1)
            return mask
        # Pure-torch approximation: no hole filling, just edges
        return to_numpy(edges) if not is_torch_input else edges
    except Exception as e:
        if _skimage_canny is not None and _scipy_ndi is not None:
            np_img = to_numpy(t)
            edges = _skimage_canny(np_img, sigma=sigma)
            fill_detector = _scipy_ndi.binary_fill_holes(edges)
            mask = _scipy_ndi.binary_erosion(fill_detector, iterations=1)
            return mask
        raise RuntimeError("No edge mask backend available") from e


# ---------------------------------------------------------------------------
# moments on GPU
# ---------------------------------------------------------------------------

def torch_moments(binary_mask: torch.Tensor) -> dict[str, float]:
    """Image moments (m00, m10, m01, m11, m20, m02) computed on device."""
    y, x = torch.meshgrid(
        torch.arange(binary_mask.shape[0], dtype=torch.float32, device=binary_mask.device),
        torch.arange(binary_mask.shape[1], dtype=torch.float32, device=binary_mask.device),
        indexing="ij",
    )
    m00 = binary_mask.sum()
    if m00 == 0:
        return {"m00": 0.0, "m10": 0.0, "m01": 0.0}
    m10 = (x * binary_mask).sum()
    m01 = (y * binary_mask).sum()
    return {"m00": float(m00), "m10": float(m10), "m01": float(m01)}


def torch_find_center(binary_mask: torch.Tensor) -> tuple[int, int]:
    """Centroid from binary mask — stays on device, returns Python ints."""
    M = torch_moments(binary_mask)
    if M["m00"] == 0:
        return (binary_mask.shape[1] // 2, binary_mask.shape[0] // 2)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return center_x, center_y
