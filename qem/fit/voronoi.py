from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import torch
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.segmentation import watershed
from tqdm import tqdm as progressbar
from scipy.spatial import cKDTree

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from qem.utils.tensors import clone_params, to_numpy, to_tensor

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def voronoi_integrate(
    image,
    points_x,
    points_y,
    method="Voronoi",
    max_radius="Auto",
    show_progressbar=True,
    remove_edge_cells=False,
    edge_pixels=1,
    pbc=False,
):
    """Given a spectrum image a set of points and a maximum outer radius,
    this function integrates around each point in an image, using either
    Voronoi cell or watershed segmentation methods.

    Parameters
    ----------
    image : 2D, 3D or 4D array-like
        Assuming 2D, 3D or 4D dataset where the spatial dimensions are 2D and
        any remaining dimensions are spectral.
    point_x, point_y : list
        Detailed list of the x and y coordinates of each point of
        interest within the image.
    method : string
        'Voronoi' or 'Watershed'
    max_radius : {'Auto'} int
        A maximum outer radius for each Voronoi Cell.
        If a pixel exceeds this radius it will not be included in the cell.
        This allows analysis of a surface and particles.
        If 'max_radius' is left as 'Auto' then it will be set to the largest
        dimension in the image.
    remove_edge_cells : bool
        Determine whether to replace the cells touching the signal edge with
        np.nan values, which makes automatic contrast estimation easier later
    edge_pixels : int
        Only used if remove_edge_cells is True. Determines the number of
        pixels from the border to remove.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    integrated_intensity : NumPy array
        An array where dimension 0 is the same length as points, and subsequent
        subsequent dimension are energy dimensions.
    intensity_record : HyperSpy signal, same size as s
        Each pixel/voxel in a particular segment or region has the value of the
        integration, value.
    point_record : 2D numpy array, same size as image
        Image showing where each integration region is, pixels equating to
        point 0 (integrated_intensity[0]) all have value 0, all pixels
        equating to integrated_intensity[1] all have value 1 etc.

    Note
    ----
    Works in principle with 3D and 4D data sets but will quickly hit a
    memory error with large sizes.

    """
    if len(image.shape) < 2:
        raise ValueError("image must have at least 2 dimensions")
    intensity_record = np.zeros_like(image, dtype=float)
    integrated_intensity = np.zeros(image.shape[:-2])
    integrated_intensity = np.stack(
        [integrated_intensity for i in range(len(points_x))]
    )

    points = np.array((points_y, points_x))
    # Setting max_radius to the width of the image, if none is set.
    if method == "Voronoi":
        if max_radius == "Auto":
            max_radius = max(image.shape[-2:])
        elif max_radius <= 0:
            raise ValueError("max_radius must be higher than 0.")
        point_record = voronoi_point_record(image, points, max_radius, pbc=pbc)

    elif method == "Watershed":
        if len(image.shape) > 2:
            raise ValueError(
                "Currently Watershed method is only implemented for 2D data."
            )
        points_map = _make_mask(image.T, points[0], points[1])
        point_record = watershed(-image, points_map.T)

    else:
        raise NotImplementedError("Oops! You have asked for an unimplemented method.")
    point_record -= 1

    def process_point(point_index):
        return point_index, get_integrated_intensity(point_record, image, point_index)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_point, point_index)
            for point_index in range(points.shape[1])
        ]
        for future in tqdm(
            as_completed(futures),
            total=points.shape[1],
            desc="Integrating",
            disable=not show_progressbar,
        ):
            point_index, intensity = future.result()
            integrated_intensity[point_index] = intensity

    for i, j in progressbar(
        np.ndindex(image.shape[-2:]),
        desc="Building intensity map",
        total=np.prod(image.shape[-2:]),
        leave=False,
    ):

        point_index = point_record[i, j]
        if point_index == -1:
            intensity_record[..., i, j] = np.nan
        else:
            summed = integrated_intensity[point_index]
            intensity_record[..., i, j] = summed

    if remove_edge_cells:
        remove_integrated_edge_cells(
            integrated_intensity,
            intensity_record,
            point_record,
            edge_pixels=edge_pixels,
            use_nans=True,
            inplace=True,
        )
    return integrated_intensity, intensity_record, point_record

def voronoi_point_record(image, points, max_radius, pbc=False, box=None):
    """Voronoi cell assignment via scipy cKDTree.

    Tested torch ``cdist + argmin`` on this — even on MPS the brute-
    force O(P·N) pairwise distance loses to scipy's O(P·log N) k-d
    tree (242ms scipy vs 447ms MPS vs 1467ms CPU torch on 256×256
    image with ~3000 atoms). k-d tree wins on this access pattern.

    Parameters
    ----------
    image : 2D array or shape tuple
        The image (or its shape) to create the Voronoi map for.
    points : array-like, shape (2, N)
        Coordinates of the points (y, x).
    max_radius : float
        Maximum radius for Voronoi cell assignment.
    pbc : bool, optional
        Whether to use periodic boundary conditions.
    box : tuple or None
        Box size for PBC. Required if pbc=True.

    Returns
    -------
    point_record : 2D numpy array
        Voronoi array where equal values belong to the same cell.
    """
    shape = image.shape if hasattr(image, "shape") else image
    points = np.asarray(points)
    if points.shape[0] != 2:
        raise ValueError("points should have shape (2, N)")
    points_xy = np.column_stack((points[0], points[1]))

    if pbc:
        if box is None:
            box = shape
        tree = cKDTree(points_xy, boxsize=box)
    else:
        tree = cKDTree(points_xy)

    grid_y, grid_x = np.indices(shape)
    grid_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

    dist, idx = tree.query(grid_points, distance_upper_bound=max_radius)
    idx[dist >= max_radius] = -1
    return idx.reshape(shape) + 1

def calculate_point_record(image, points, max_radius, pbc=False):
    """
    Creates a Voronoi array where equal values belong to
    the same Voronoi cell
    
    This is slow version from atomap

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    points: Array like of shape (2,N)
    max_radius: Integer, max radius of each Voronoi Cell

    Returns
    -------
    point_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    point_record = np.zeros(image.shape[-2:], dtype=int)
    for i, j in progressbar(
        np.ndindex(point_record.shape),
        desc="Calculating Voronoi",
        total=np.prod(point_record.shape),
        leave=False,
    ):
        min_index, dist_min = find_smallest_distance(
            i, j, points, image_shape=image.shape, pbc=pbc
        )
        if dist_min >= max_radius:
            point_record[i][j] = 0
        else:
            point_record[i][j] = min_index + 1
    return point_record

def get_integrated_intensity(point_record, image, point_index, include_edge_cells=True):
    """
    Using a Voronoi point_record array, integrate a (minimum 2D)
    image array at each pixel

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    image : The ndarray to integrate the voronoi cells on
    point_index: Array like of shape (2,N)

    Returns
    -------
    integrated_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    currentMask = point_record == point_index
    currentFeature = currentMask * image
    integrated_record = np.sum(currentFeature, axis=(-1, -2))
    return integrated_record


@nb.jit()
def find_smallest_distance(i, j, points, image_shape=None, pbc=False):
    """
    Finds the smallest distance between coordinates (i, j)
    and a list of coordinates.

    Parameters
    ----------
    i : Integer
    j : Integer
    points : array like of shape (2,N)

    Returns
    -------
    distMin  : Minimum distance
    minIndex : Index of minimum distance in points

    Example
    -------
    >>> import numpy as np
    >>> points = np.random.random((2, 10000))
    >>> i, j = 0.5, 0.5
    >>> smallest_distance = find_smallest_distance(i, j, points)

    """
    if pbc:
        height, width = image_shape
        distance_log = np.inf * np.ones(points.shape[1])
        for k in range(points.shape[1]):
            dx = min(abs(points[0, k] - i), width - abs(points[0, k] - i))
            dy = min(abs(points[1, k] - j), height - abs(points[1, k] - j))
            distance_log[k] = (dx**2 + dy**2) ** 0.5
    else:
        distance_log = (
            (points[0] - float(i)) ** 2 + (points[1] - float(j)) ** 2
        ) ** 0.5
    minIndex = np.argmin(distance_log)
    distMin = distance_log[minIndex]
    return minIndex, distMin


def remove_integrated_edge_cells(
    i_points, i_record, p_record, edge_pixels=1, use_nans=True, inplace=False
):
    """Removes any cells that touch within a number of pixels of
    the image border.

    Note on using use_nans: If this is used on a dataset with more than
    two dimensions, the resulting HyperSpy i_record signal might be needed to
    be viewed with i_record.plot(navigator='slider'), since HyperSpy may throw
    an error when plotting a dataset with only NaNs present.

    Parameters
    ----------
    i_points : NumPy array
        The output of the Atomap integrate function or method
    i_record : HyperSpy signal
        The output of the Atomap integrate function or method
    p_record : HyperSpy signal
        The output of the Atomap integrate function or method

    Returns
    -------
    i_points : NumPy array
        Modified list of integrated intensities with either np.nan or 0
        on the removed values, which preserves the atom index.
    i_record : HyperSpy signal
        Modified integrated intensity record, with either np.nan or 0
        on the removed values, which preserves the atom index
    p_record : HyperSpy signal, same size as image
        Modified points record, where removed areas have value = -1.

    Example
    -------

    >>> s = am.dummy_data.get_fantasite()
    >>> points_x, points_y = am.get_atom_positions(s).T
    >>> i, ir, pr = am.integrate(
    ...    s,
    ...    points_x,
    ...    points_y,
    ...    method='Voronoi',
    ...    remove_edge_cells=False)
    >>> from atomap.tools import remove_integrated_edge_cells
    >>> i2, ir2, pr2 = remove_integrated_edge_cells(
    ...    i, ir, pr, edge_pixels=5, use_nans=True)

    """
    if not inplace:
        i_points = i_points.copy()
        i_record = i_record.deepcopy()
        p_record = p_record.deepcopy()

    border = _border_elems(p_record.data, edge_pixels)
    border_indices = np.array(list(set(border)))
    indices = np.in1d(p_record.data, border_indices)
    indices = indices.reshape(p_record.data.shape)
    i_points[border_indices] = np.nan if use_nans else 0
    i_record.data[..., indices] = np.nan if use_nans else 0
    p_record.data[indices] = -1

    if not inplace:
        return i_points, i_record, p_record


def _make_mask(image, points_x, points_y):
    """
    Create points_map for the watershed integration
    function
    """
    mask = np.zeros(image.shape[-2:])
    indices = np.round(np.array([points_y, points_x])).astype(int)
    values = np.arange(len(points_x))
    mask[tuple(indices)] = values
    return mask


def _border_elems(image, pixels=1):
    """
    Return the values of the edges along the border of the image, with
    border width `pixels`.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([
    ...     [1,1,1],
    ...     [2,5,3],
    ...     [4,4,4]])
    >>> b = _border_elems(a, pixels=1)

    """
    arr = np.ones_like(image, dtype=bool)
    arr[pixels: -1 - (pixels - 1), pixels: -1 - (pixels - 1)] = False
    return image[arr]


def _batched_gaussian_lm(
    crops: torch.Tensor,
    masks: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    bg: torch.Tensor,
    *,
    max_iter: int = 15,
    tol: float = 1e-5,
    damping_init: float = 1e-2,
    max_position_drift: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-atom Levenberg-Marquardt for 2-D Gaussian fits, batched.

    ``crops`` / ``masks`` are ``(N, k, k)`` tensors, one cell per atom.
    All other arguments are ``(N,)`` per-atom parameter vectors. Returns
    refined ``(px, py)`` only — heights/widths/bg are nuisance vars.

    The Jacobian is computed analytically, so the inner step is just
    one batched matmul + one batched ``torch.linalg.solve`` on
    ``(N, 5, 5)`` systems. This is the block-diagonal Newton step that
    a generic ``LBFGS`` or ``LBFGS-B`` cannot exploit — and it's why
    those generic batched torch optimizers fail on this problem
    shape.
    """
    device = crops.device
    N = crops.shape[0]
    eps = 1e-12

    # Anchor positions: never wander more than max_position_drift from
    # the COM init. STEM cells are small and the COM is unbiased; the
    # LM step is meant to refine, not relocate.
    px_anchor = px.clone()
    py_anchor = py.clone()

    # Mutable copies (not nn.Parameter — we manage updates ourselves).
    px = px.clone()
    py = py.clone()
    h = h.clone().clamp(min=eps)
    w = w.clone().clamp(min=0.5)
    bg = bg.clone()
    damping = torch.full((N,), damping_init, device=device)

    for _ in range(max_iter):
        # Forward + analytic Jacobian.
        dx = x_grid[None, ...] - px[:, None, None]      # (N, k, k)
        dy = y_grid[None, ...] - py[:, None, None]
        r2 = dx * dx + dy * dy
        w2 = (w * w)[:, None, None]
        env = torch.exp(-r2 / (2.0 * w2))                # (N, k, k)
        g0 = h[:, None, None] * env                      # peak (no bg)
        pred = g0 + bg[:, None, None]
        residual = (pred - crops) * masks                # masked diff
        # Per-pixel Jacobian, position-only (h, w, bg held fixed at
        # their data-driven init): this is a strict refinement of the
        # 2-D position. Jacobian shape: (N, k, k, 2).
        J_px = g0 * dx / w2
        J_py = g0 * dy / w2
        J = torch.stack([J_px, J_py], dim=-1) * masks[..., None]
        J_flat = J.reshape(N, -1, 2)
        r_flat = residual.reshape(N, -1)

        # Normal equations per atom: (JᵀJ + λI) δ = Jᵀ r
        JtJ = J_flat.transpose(-2, -1) @ J_flat            # (N, 2, 2)
        Jtr = (J_flat.transpose(-2, -1) @ r_flat[..., None]).squeeze(-1)
        eye = torch.eye(2, device=device).expand(N, 2, 2)
        A = JtJ + damping[:, None, None] * eye
        try:
            delta = torch.linalg.solve(A, Jtr[..., None]).squeeze(-1)
        except RuntimeError:
            damping = damping * 10.0
            continue

        # Tentative update (Newton step in -delta direction). Clamp
        # positions to stay within max_position_drift of the COM anchor.
        new_px = (px - delta[:, 0]).clamp(
            min=px_anchor - max_position_drift,
            max=px_anchor + max_position_drift,
        )
        new_py = (py - delta[:, 1]).clamp(
            min=py_anchor - max_position_drift,
            max=py_anchor + max_position_drift,
        )
        new_h = h
        new_w = w
        new_bg = bg

        # Per-atom accept / reject by loss.
        old_loss = (residual * residual).sum(dim=(1, 2))
        n_dx = x_grid[None, ...] - new_px[:, None, None]
        n_dy = y_grid[None, ...] - new_py[:, None, None]
        n_pred = new_h[:, None, None] * torch.exp(
            -(n_dx * n_dx + n_dy * n_dy) / (2.0 * (new_w * new_w)[:, None, None])
        ) + new_bg[:, None, None]
        new_loss = ((n_pred - crops) * masks).pow(2).sum(dim=(1, 2))
        accept = new_loss < old_loss

        # LM damping update: halve on accept, ×4 on reject.
        damping = torch.where(accept, damping * 0.5, damping * 4.0).clamp(
            min=1e-7, max=1e7,
        )

        ax = accept[..., None] if False else accept
        px = torch.where(ax, new_px, px)
        py = torch.where(ax, new_py, py)
        h = torch.where(ax, new_h, h)
        w = torch.where(ax, new_w, w)
        bg = torch.where(ax, new_bg, bg)

        # Convergence: if no per-atom delta exceeds tol, stop.
        max_step = delta.abs().amax(dim=-1)
        if torch.all(max_step < tol):
            break

    return px, py


def _fit_voronoi_batched(
    self,
    params: dict | None = None,
    max_radius: int | float | None = None,
    tol: float = 1e-5,
    border: int = 0,
    refine: bool = False,
    max_iter: int = 15,
):
    """Batched Voronoi position refinement — all atoms in one torch call.

    Computes a closed-form per-cell centroid (first moment of
    ``crops * masks``) for every atom, then accepts each candidate
    position only if it strictly decreases the per-cell SSE under the
    GLOBAL Gaussian model (using each atom's current ``height`` /
    ``width``). This guard mirrors the legacy curve_fit path's
    implicit "fall back to p0 on failure" behaviour while running in
    one batched torch op.

    Set ``refine=True`` to additionally run a per-atom Levenberg-
    Marquardt step on (px, py) with global (h, w) frozen. LM is
    correct in principle but on data where ``fit_stochastic`` already
    converged, it tends to over-fit each cell and hurt the global
    residual. Useful for problems where positions are coarsely
    initialised and need genuine refinement.
    """
    if params is None:
        if self.params is not None and "pos_x" in self.params and "pos_y" in self.params:
            params = self.params
        else:
            params = self.init_params()

    pos_x_t = params["pos_x"] if torch.is_tensor(params["pos_x"]) else torch.as_tensor(params["pos_x"])
    pos_y_t = params["pos_y"] if torch.is_tensor(params["pos_y"]) else torch.as_tensor(params["pos_y"])
    width_t = params["width"] if torch.is_tensor(params["width"]) else torch.as_tensor(params["width"])

    if max_radius is None:
        max_radius_t = width_t.detach().max() * 3.0
        max_radius = float(max_radius_t.item())
    max_radius = int(max(1, round(float(max_radius))))
    k = 2 * max_radius + 1

    # Build the voronoi point_record on numpy (cKDTree is fast).
    image_np = self.image if isinstance(self.image, np.ndarray) else to_numpy(self.image)
    coords_np = np.stack([to_numpy(pos_y_t), to_numpy(pos_x_t)])
    point_record = voronoi_point_record(image_np, coords_np, max_radius)

    H, W = image_np.shape
    N = pos_x_t.shape[0]

    # Atom-centred bounding boxes (clipped to image).
    pos_x_np = to_numpy(pos_x_t)
    pos_y_np = to_numpy(pos_y_t)
    cx = np.clip(np.round(pos_x_np).astype(np.int64), 0, W - 1)
    cy = np.clip(np.round(pos_y_np).astype(np.int64), 0, H - 1)
    x0 = np.clip(cx - max_radius, 0, W - 1)
    x1 = np.clip(cx + max_radius + 1, 1, W)
    y0 = np.clip(cy - max_radius, 0, H - 1)
    y1 = np.clip(cy + max_radius + 1, 1, H)
    # Where each window starts after clipping (relative to atom centre).
    x_off = cx - max_radius   # may be negative; we clip below
    y_off = cy - max_radius

    # Vectorised (N, k, k) crops + masks build via scatter. Replaces a
    # Python loop that did `point_record == i + 1` once per atom —
    # O(P·N) full-image scans → O(P) one pass over assigned pixels.
    crops = np.zeros((N, k, k), dtype=np.float32)
    masks = np.zeros((N, k, k), dtype=bool)
    local_min = np.full(N, np.inf, dtype=np.float32)

    # All assigned pixel-atom pairs (point_record stores atom_id+1, 0=bg).
    assigned = point_record - 1
    pix_y, pix_x = np.where(assigned >= 0)
    atom_ids = assigned[pix_y, pix_x]
    pix_vals = image_np[pix_y, pix_x].astype(np.float32)

    # Per-pixel destination in the (k, k) window.
    dst_y_pix = pix_y - y_off[atom_ids]
    dst_x_pix = pix_x - x_off[atom_ids]
    in_win = (
        (dst_y_pix >= 0) & (dst_y_pix < k)
        & (dst_x_pix >= 0) & (dst_x_pix < k)
    )
    if not np.any(in_win):
        return self.params if self.params is not None else params
    pix_y = pix_y[in_win]
    pix_x = pix_x[in_win]
    atom_ids = atom_ids[in_win]
    pix_vals = pix_vals[in_win]
    dst_y_pix = dst_y_pix[in_win]
    dst_x_pix = dst_x_pix[in_win]

    masks[atom_ids, dst_y_pix, dst_x_pix] = True
    # Per-atom min via vectorised np.minimum.at (unbuffered ufunc).
    np.minimum.at(local_min, atom_ids, pix_vals)
    crops[atom_ids, dst_y_pix, dst_x_pix] = pix_vals - local_min[atom_ids]
    # An atom is "valid" iff at least one pixel landed in its window.
    valid = np.isfinite(local_min)
    local_min = np.where(valid, local_min, 0.0)

    if border > 0:
        # Exclude atoms whose bbox touches the image border.
        edge = (
            (cx < border) | (cx > W - border)
            | (cy < border) | (cy > H - border)
        )
        valid &= ~edge

    if not valid.any():
        # Nothing to refine.
        return self.params if self.params is not None else params

    device = self.device
    crops_t = torch.as_tensor(crops, device=device)
    masks_t = torch.as_tensor(masks, device=device, dtype=torch.float32)
    valid_t = torch.as_tensor(valid, device=device, dtype=torch.float32)
    x_off_t = torch.as_tensor(x_off, device=device, dtype=torch.float32)
    y_off_t = torch.as_tensor(y_off, device=device, dtype=torch.float32)

    # Window-relative grid (shared across all atoms).
    win = torch.arange(k, dtype=torch.float32, device=device)
    x_grid_w, y_grid_w = torch.meshgrid(win, win, indexing="xy")  # (k, k)

    # Initial per-cell positions in window coords (atom centre at
    # max_radius, possibly offset because the bbox was clipped).
    init_px_w = torch.as_tensor(pos_x_np, device=device, dtype=torch.float32) - x_off_t
    init_py_w = torch.as_tensor(pos_y_np, device=device, dtype=torch.float32) - y_off_t

    # Per-atom GLOBAL model height + width (the values the rest of
    # qem.fit uses). The acceptance guard below evaluates the global
    # model's local fit at each atom — moving an atom is only useful
    # if it improves THAT contribution.
    h_global = params["height"] if torch.is_tensor(params["height"]) else torch.as_tensor(params["height"])
    h_global = h_global.detach().to(device=device, dtype=torch.float32)
    if h_global.ndim == 0:
        h_global = h_global.expand(N)

    width_param = width_t.detach().to(device=device, dtype=torch.float32)
    if width_param.numel() == 1 or width_param.numel() == int(getattr(self, "num_atom_types", 1)):
        atom_types = self.atom_types if isinstance(self.atom_types, np.ndarray) else to_numpy(self.atom_types)
        if len(atom_types) == N:
            w_global = width_param[
                torch.as_tensor(atom_types, device=device, dtype=torch.long)
            ]
        else:
            w_global = width_param.expand(N) if width_param.numel() == 1 else width_param[:N]
    else:
        w_global = width_param[:N]
    w_global = w_global.clamp(min=0.5)

    # Stage 1 — closed-form COM as the candidate update.
    weights = (crops_t * masks_t).clamp(min=0.0)            # (N, k, k)
    total = weights.sum(dim=(1, 2)).clamp(min=1e-6)         # (N,)
    com_px = (weights * x_grid_w[None, ...]).sum(dim=(1, 2)) / total
    com_py = (weights * y_grid_w[None, ...]).sum(dim=(1, 2)) / total

    if refine:
        # Stage 2 — per-atom LM, position-only with h, w fixed at
        # global values. Init at COM so the LM nudges from a good
        # starting point.
        bg_init = torch.zeros_like(h_global)
        new_px, new_py = _batched_gaussian_lm(
            crops_t, masks_t, x_grid_w, y_grid_w,
            com_px.clone(), com_py.clone(),
            h_global, w_global, bg_init,
            max_iter=max_iter, tol=tol,
        )
        candidate_px, candidate_py = new_px, new_py
    else:
        candidate_px, candidate_py = com_px, com_py

    # Acceptance guard against the GLOBAL Gaussian model — the only
    # objective that actually maps to a residual improvement. Move
    # an atom only if its candidate position fits the cell pixels
    # better than the existing position, using THIS atom's global
    # (h, w) parameters.
    def _global_cell_loss(px_t, py_t):
        dx = x_grid_w[None, ...] - px_t[:, None, None]
        dy = y_grid_w[None, ...] - py_t[:, None, None]
        gauss = h_global[:, None, None] * torch.exp(
            -(dx * dx + dy * dy) / (2.0 * (w_global[:, None, None] ** 2))
        )
        return ((gauss - crops_t) * masks_t).pow(2).sum(dim=(1, 2))

    old_loss = _global_cell_loss(init_px_w, init_py_w)
    new_loss = _global_cell_loss(candidate_px, candidate_py)
    accept = new_loss < old_loss
    px_w = torch.where(accept, candidate_px, init_px_w)
    py_w = torch.where(accept, candidate_py, init_py_w)

    px_opt = px_w + x_off_t
    py_opt = py_w + y_off_t

    # Only update positions for atoms with valid cells (matching legacy).
    pos_x_arr = to_numpy(pos_x_t).astype(np.float32)
    pos_y_arr = to_numpy(pos_y_t).astype(np.float32)
    px_np = to_numpy(px_opt)
    py_np = to_numpy(py_opt)
    pos_x_arr[valid] = px_np[valid]
    pos_y_arr[valid] = py_np[valid]

    out_params = clone_params(params)
    out_params["pos_x"] = torch.as_tensor(pos_x_arr, dtype=torch.float32, device=device)
    out_params["pos_y"] = torch.as_tensor(pos_y_arr, dtype=torch.float32, device=device)
    self.params = out_params
    return out_params


def fit_voronoi(
    self,
    params: dict = None,  # initial params, optional
    max_radius: int = None,  # optional, for Voronoi cell size
    tol: float = 1e-3,
    border: int = 0,  # optional, exclude border pixels
    refine: bool = False,    # extra per-atom Levenberg-Marquardt
):
    """Refine atomic positions to per-cell Gaussian centroids.

    Closed-form per-cell COM with a global-model acceptance guard
    (a candidate position is kept only if it improves that atom's
    contribution to the global residual). All N cells in a single
    batched torch op. Set ``refine=True`` for an extra per-atom
    Levenberg-Marquardt step (custom batched torch impl, useful when
    positions are coarsely initialised).
    """
    return _fit_voronoi_batched(
        self, params=params, max_radius=max_radius, tol=tol,
        border=border, refine=refine,
    )

def voronoi_integration(self, max_radius: float = None, plot=False,save=False):
    """
    Compute the Voronoi integration of the atomic columns.

    Returns:
        np.array: The Voronoi integration of the atomic columns.
    """
    if self.params is None:
        raise ValueError("Please initialize the parameters first.")
    if self.fit_background:
        image = (self.image - to_numpy(self.params["background"]))
    else:
        image = (self.image - self.init_background)
    pos_x = self.params["pos_x"]
    pos_y = self.params["pos_y"]
    pos_x = to_numpy(pos_x)
    pos_y = to_numpy(pos_y)
    if max_radius is None:
        max_radius = self.params["width"].max() * 5
        max_radius = to_numpy(max_radius)
    integrated_intensity, intensity_record, point_record = voronoi_integrate(
        image, pos_x, pos_y, max_radius=max_radius, pbc=self.pbc
    )
    integrated_intensity = integrated_intensity * self.dx**2
    intensity_record = intensity_record * self.dx**2
    self._voronoi_volume = integrated_intensity
    self._voronoi_map = intensity_record
    self._voronoi_cell = point_record
    if plot:
        plt.imshow(intensity_record, cmap="viridis")
        plt.colorbar(label="Voronoi Integrated Intensity")
    if save:
        plt.savefig("Voronoi Integrated Intensity.png", dpi=300)
        plt.savefig("Voronoi Integrated Intensity.svg")

    return integrated_intensity, intensity_record, point_record

# parameters updates and convergence



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.fit_voronoi = fit_voronoi
    cls._fit_voronoi_batched = _fit_voronoi_batched
    cls.voronoi_integration = voronoi_integration


__all__ = [
    "fit_voronoi",
    "voronoi_integration",
    "_bind",
]

