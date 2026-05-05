import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

from qem.io.statstem import read_statstem
from qem.fit.fitter import Fitter
from qem.utils.tensors import to_numpy


def goodness_of_fit(image, prediction, *, noise_var=None):
    """Quick goodness-of-fit summary for a fitted prediction.

    Args:
        image: ground-truth observation (H, W).
        prediction: model prediction (H, W).
        noise_var: per-pixel noise variance estimate. If None, the median
            of the residual variance over a 16-pixel-wide flat border is
            used as a proxy (assumes that border is signal-free).

    Returns a dict with:
        L2_std, L1_mean, L2_max — residual norms
        chi2_red — reduced chi-squared (target ≈ 1 if noise_var is right)
        psd_white_ratio — fraction of residual 2D power spectrum within
            ±20% of the median (1.0 means perfectly white residuals;
            <1.0 means structured residuals — model still missing signal)
        psd_peak_freq — radial frequency (cycles/px) of the strongest
            non-DC peak in the residual PSD; 0 if no significant peak.
    """
    image = np.asarray(image, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    res = image - prediction

    out = {
        "L2_std": float(np.std(res)),
        "L1_mean": float(np.mean(np.abs(res))),
        "L2_max": float(np.max(np.abs(res))),
    }

    # Reduced chi-squared. If the user didn't pass a noise variance, fall
    # back to a flat-border estimate.
    if noise_var is None:
        b = 16
        border = np.concatenate([
            res[:b].ravel(), res[-b:].ravel(),
            res[b:-b, :b].ravel(), res[b:-b, -b:].ravel(),
        ])
        noise_var = float(np.var(border)) if border.size else float(np.var(res))
    noise_var = max(float(noise_var), 1e-12)
    dof = max(res.size, 1)
    out["chi2_red"] = float(np.sum(res * res) / (dof * noise_var))

    # 2D residual power spectrum (FFT, drop DC).
    spec = np.fft.fft2(res)
    psd = np.abs(spec) ** 2
    psd_flat = psd.ravel()
    psd_flat[0] = 0.0  # drop DC
    psd_no_dc = psd_flat[psd_flat > 0]
    if psd_no_dc.size:
        med = float(np.median(psd_no_dc))
        within_band = np.sum(
            (psd_no_dc >= 0.8 * med) & (psd_no_dc <= 1.2 * med),
        )
        out["psd_white_ratio"] = float(within_band / psd_no_dc.size)
    else:
        out["psd_white_ratio"] = 1.0

    # Radial average of the PSD; locate dominant non-DC peak.
    h, w = res.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx * fx + fy * fy).ravel()
    r_bins = np.linspace(0, 0.5, 64)
    idx = np.digitize(r, r_bins) - 1
    radial = np.zeros(len(r_bins))
    counts = np.zeros(len(r_bins))
    for i, p in zip(idx, psd_flat):
        if 0 <= i < len(r_bins):
            radial[i] += p
            counts[i] += 1
    radial = radial / np.maximum(counts, 1)
    radial[0] = 0.0  # ignore DC bin
    nonzero = radial[radial > 0]
    median_radial = float(np.median(nonzero)) if nonzero.size else 1.0
    if radial.max() > 2 * median_radial:
        out["psd_peak_freq"] = float(r_bins[int(np.argmax(radial))])
    else:
        out["psd_peak_freq"] = 0.0

    return out


def _flat_border_noise_var(residual: np.ndarray, border: int = 16) -> float:
    """Estimate per-pixel noise variance from a flat border of the image.

    Assumes the border is signal-free (no atoms / no particle). Returns
    the variance of those pixels — a Gaussian-noise proxy that's
    independent of how well the centre is fit.
    """
    if residual.shape[0] <= 2 * border or residual.shape[1] <= 2 * border:
        return float(np.var(residual))
    pixels = np.concatenate([
        residual[:border].ravel(), residual[-border:].ravel(),
        residual[border:-border, :border].ravel(),
        residual[border:-border, -border:].ravel(),
    ])
    return float(np.var(pixels))


def crlb_per_atom(fitter, *, noise_var: float | None = None) -> dict:
    """Per-atom Cramér-Rao lower bound on (x, y, h, SCS).

    Closed-form CRLB for an isolated isotropic 2D Gaussian peak with
    Gaussian pixel noise of variance ``σ²_n``. The Fisher information
    block is diagonal by symmetry::

        F_xx = F_yy = π·h² / (2·σ²_n)
        F_hh         = π·w² / σ²_n

    so the lower-bound standard deviations are::

        σ(x) = σ(y) = σ_n / (h · √(π/2))     [pixel units]
        σ(h)         = σ_n / (w · √π)
        σ(SCS)/SCS   = √( (σ(h)/h)² + (2·σ(w)/w)² )   (here w shared
                       and considered exact ⇒ σ(SCS)/SCS ≈ σ(h)/h)

    Args:
        fitter: a fitted :class:`qem.fit.fitter.Fitter`.
        noise_var: per-pixel noise variance. If ``None``, estimated from
            a flat border of the residual image.

    Returns:
        dict with arrays per atom (in pixel/intensity units, matching
        Fitter parameter conventions) and aggregate scalars:

        - ``sigma_x``, ``sigma_y``: position CRLB in pixels.
        - ``sigma_x_ang``, ``sigma_y_ang``: same in Å.
        - ``sigma_h``: height CRLB.
        - ``rel_sigma_h``: σ(h) / h (relative).
        - ``rel_sigma_scs``: σ(SCS) / SCS (relative, ≈ rel_sigma_h
          when width is treated as fixed).
        - ``noise_var``: variance used.
    """
    image = np.asarray(fitter.image, dtype=np.float64)
    pred = np.asarray(fitter.prediction, dtype=np.float64)
    res = image - pred

    if noise_var is None:
        noise_var = _flat_border_noise_var(res)
    sigma_n = float(np.sqrt(max(noise_var, 1e-12)))

    h = to_numpy(fitter.params["height"]).astype(np.float64)
    w_raw = to_numpy(fitter.params["width"]).astype(np.float64)
    if getattr(fitter, "same_width", True):
        atom_types = to_numpy(fitter.params["atom_types"]).astype(np.int64)
        w = w_raw[atom_types]
    else:
        w = w_raw

    # Closed-form CRLB. Guard against h = 0 (atoms that converged to zero
    # amplitude have no positional information).
    h_safe = np.maximum(np.abs(h), 1e-6)
    sigma_x = sigma_n / (h_safe * np.sqrt(np.pi / 2.0))
    sigma_h = sigma_n / (np.maximum(w, 1e-6) * np.sqrt(np.pi))

    return {
        "sigma_x": sigma_x,
        "sigma_y": sigma_x.copy(),
        "sigma_x_ang": sigma_x * fitter.dx,
        "sigma_y_ang": sigma_x * fitter.dx,
        "sigma_h": sigma_h,
        "rel_sigma_h": sigma_h / h_safe,
        "rel_sigma_scs": sigma_h / h_safe,
        "noise_var": float(noise_var),
        "sigma_n": sigma_n,
    }


def residual_per_atom(
    fitter,
    *,
    window_factor: float = 3.0,
    noise_var: float | None = None,
) -> dict:
    """Per-atom residual quality inside a local window around each peak.

    For each atom the residual is summarised inside a square window of
    half-width ``window_factor · w`` (3σ by default — covers >99% of the
    peak). Returns arrays of per-atom statistics suitable for spotting
    locally mis-fit atoms (the kind of structured residual that drives
    ``chi2_red`` above 1 and makes the residual PSD non-white).

    Returns:
        dict with arrays per atom and an aggregate noise-variance:

        - ``res_std``: residual standard deviation in the window.
        - ``res_l1``:  mean absolute residual in the window.
        - ``res_max``: max absolute residual.
        - ``res_sum``: signed sum of residual (positive ⇒ peak under-
          predicted, negative ⇒ over-predicted).
        - ``chi2_red``: local reduced χ², =⟨r²⟩ / σ²_n. Atoms with
          chi2_red ≫ 1 are mis-fit.
        - ``window_size``: side length of the window in pixels.
    """
    image = np.asarray(fitter.image, dtype=np.float64)
    pred = np.asarray(fitter.prediction, dtype=np.float64)
    res = image - pred
    H, W = res.shape

    if noise_var is None:
        noise_var = _flat_border_noise_var(res)
    noise_var = max(float(noise_var), 1e-12)

    pos_x = to_numpy(fitter.params["pos_x"]).astype(np.float64)
    pos_y = to_numpy(fitter.params["pos_y"]).astype(np.float64)
    w_raw = to_numpy(fitter.params["width"]).astype(np.float64)
    if getattr(fitter, "same_width", True):
        atom_types = to_numpy(fitter.params["atom_types"]).astype(np.int64)
        w = w_raw[atom_types]
    else:
        w = w_raw

    n = pos_x.shape[0]
    res_std = np.zeros(n)
    res_l1 = np.zeros(n)
    res_max = np.zeros(n)
    res_sum = np.zeros(n)
    chi2_red = np.zeros(n)
    win_sizes = np.zeros(n, dtype=np.int32)

    # Per-atom local window. Atoms within ``window_factor·w`` pixels of
    # the image edge get a clipped window — fine for diagnostics.
    for i in range(n):
        half = max(int(window_factor * float(w[i])), 1)
        x_lo = max(int(pos_x[i] - half), 0)
        x_hi = min(int(pos_x[i] + half) + 1, W)
        y_lo = max(int(pos_y[i] - half), 0)
        y_hi = min(int(pos_y[i] + half) + 1, H)
        block = res[y_lo:y_hi, x_lo:x_hi]
        if block.size == 0:
            continue
        win_sizes[i] = block.shape[0] * block.shape[1]
        res_std[i] = float(np.std(block))
        res_l1[i] = float(np.mean(np.abs(block)))
        res_max[i] = float(np.max(np.abs(block)))
        res_sum[i] = float(np.sum(block))
        chi2_red[i] = float(np.mean(block * block) / noise_var)

    return {
        "res_std": res_std,
        "res_l1": res_l1,
        "res_max": res_max,
        "res_sum": res_sum,
        "chi2_red": chi2_red,
        "window_size": win_sizes,
        "noise_var": float(noise_var),
    }


def fit_efficiency(
    fitter,
    *,
    noise_var: float | None = None,
    window_factor: float = 3.0,
) -> dict:
    """Compare per-atom residual to per-atom CRLB.

    A statistically efficient estimator should leave residual variance
    inside each peak window ≈ σ²_n (the noise floor). The ratio of
    observed local-residual variance to ``σ²_n`` is the local
    reduced-χ². Persistent ratios > 1 signal model insufficiency
    (shared width too coarse, missed peaks, wrong peak shape) rather
    than optimizer failure — CRLB is satisfied only when the model is
    correctly specified.

    Returns:
        dict combining ``crlb_per_atom`` and ``residual_per_atom`` plus:

        - ``efficiency_x``: 1.0 if local-residual is at noise floor
          AND fit reached CRLB on x; otherwise diagnostic.
        - ``frac_atoms_above_chi2``: fraction of atoms with local
          chi2_red > 2.
        - ``mean_local_chi2``: mean per-atom chi2_red.
    """
    crlb = crlb_per_atom(fitter, noise_var=noise_var)
    rpa = residual_per_atom(
        fitter, window_factor=window_factor, noise_var=crlb["noise_var"],
    )
    return {
        **{f"crlb_{k}": v for k, v in crlb.items()},
        **{f"local_{k}": v for k, v in rpa.items()},
        "frac_atoms_above_chi2": float(np.mean(rpa["chi2_red"] > 2.0)),
        "mean_local_chi2": float(np.mean(rpa["chi2_red"])),
    }


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(
            f"Method {func.__name__!r} executed in {(end_time - start_time):.4f} seconds"
        )
        return result

    return wrapper


def get_coordinates(StatSTEM):
    # Check for 'coordinates' or 'Coordinates' directly
    if "coordinates" in StatSTEM:
        coordinates = StatSTEM["coordinates"]
    elif "Coordinates" in StatSTEM:
        coordinates = StatSTEM["Coordinates"]
    # If direct coordinates are not found, check for 'BetaX' and 'BetaY'
    elif "BetaX" in StatSTEM and "BetaY" in StatSTEM:
        BetaX = StatSTEM["BetaX"]
        BetaY = StatSTEM["BetaY"]
        coordinates = np.array([BetaX, BetaY]).T
    else:
        raise ValueError("Coordinate keys not found in inputStatSTEM dictionary.")
    return coordinates


def get_scs(StatSTEM):
    if "volumes" in StatSTEM:
        scs = StatSTEM["volumes"]
    elif "Volumes" in StatSTEM:
        scs = StatSTEM["Volumes"]
    else:
        raise ValueError("Volume keys not found in inputStatSTEM dictionary.")
    return scs


class Benchmark:
    def __init__(self, filepath):
        legacyStatSTEM = read_statstem(filepath)
        try:
            if "dx" in legacyStatSTEM.keys():
                self.dx = legacyStatSTEM["dx"]
            elif "dx" in legacyStatSTEM["input"].keys():
                self.dx = legacyStatSTEM["input"]["dx"]
            if "input" in legacyStatSTEM.keys():
                inputStatSTEM = legacyStatSTEM["input"]
                self.input_coordinates = get_coordinates(inputStatSTEM)
                self.image = inputStatSTEM["obs"]
            if "output" in legacyStatSTEM.keys():
                outputStatSTEM = legacyStatSTEM["output"]
                self.output_coordinates = get_coordinates(outputStatSTEM)
                if "model" in outputStatSTEM.keys():
                    self.model_statstem = outputStatSTEM["model"]
                if (
                    "volumes" in outputStatSTEM.keys()
                    or "Volumes" in outputStatSTEM.keys()
                ):
                    self.scs_statstem = get_scs(outputStatSTEM)
            if "obs" in legacyStatSTEM.keys():
                self.image = legacyStatSTEM["obs"]
            if "coordinates" in legacyStatSTEM.keys():
                self.input_coordinates = legacyStatSTEM["coordinates"]
            if "model" in legacyStatSTEM.keys():
                self.model_statstem = legacyStatSTEM["model"]
        except:
            raise ValueError(
                "InputStatSTEM dictionary does not have correct keys in the input file."
            )

    @time_it
    def refine(
        self,
        atom_size=0.7,
        guess_radius=False,
        tol=1e-2,
        maxiter=50,
        step_size=1e-2,
        num_epoch=10,
        batch_size=1000,
        plot=True,
        # Pipeline stage flags. Defaults match the recommended pipeline
        # in :func:`qem.fit.pipeline.fit_pipeline` — per-atom VarPro
        # (StatSTEM-equivalent) plus Marquardt LM polish. Subpixel
        # refinement is OFF by default; on low-contrast images the
        # parabolic fit on a 3×3 patch is dominated by noise and can
        # displace ~25% of atoms in arbitrary directions even with the
        # Hessian sign check. Enable explicitly with ``subpixel=True``
        # when peaks are well above noise.
        width_first: bool = True,
        subpixel: bool = False,
        subpixel_window: int = 0,
        per_atom_varpro: bool = True,
        varpro_max_iter: int = 30,
        varpro_alpha: float = 0.5,
        fit_stochastic: bool = False,
        lm_polish: bool = True,
        stochastic_optimizer: str = "adam",
        stochastic_optimizer_kwargs: dict | None = None,
        lm_loss: str = "l2",
    ) -> None:
        """Run the recommended fit pipeline against the StatSTEM input image.

        Delegates to :func:`qem.fit.pipeline.fit_pipeline`, which is
        StatSTEM-equivalent (per-atom Variable Projection + LM polish
        with Marquardt diagonal scaling). The legacy stochastic-Adam
        path remains reachable via ``fit_stochastic=True,
        per_atom_varpro=False``.

        ``stochastic_optimizer`` accepts any name that
        :func:`qem.fit.loop.make_optimizer` resolves — Adam / AdamW /
        SGD / LBFGS plus anything in ``pytorch_optimizer`` (kozistr) or
        ``torch_optimizer`` (jettify): Ranger, Lion, MADGRAD, AdaBelief,
        NovoGrad, AccSGD, PID, QHAdam, Apollo, etc.
        """
        model = Fitter(self.image, dx=self.dx)
        model.coordinates = self.input_coordinates / self.dx
        model.fit_pipeline(
            atom_size=atom_size,
            subpixel=subpixel,
            subpixel_window=subpixel_window,
            width_first=width_first,
            per_atom_varpro=per_atom_varpro,
            varpro_max_iter=varpro_max_iter,
            varpro_alpha=varpro_alpha,
            stochastic=fit_stochastic,
            num_epoch=num_epoch,
            batch_size=batch_size,
            stochastic_maxiter=maxiter,
            stochastic_step_size=step_size,
            stochastic_tol=tol,
            stochastic_optimizer=stochastic_optimizer,
            stochastic_optimizer_kwargs=stochastic_optimizer_kwargs,
            lm_polish=lm_polish,
            lm_loss=lm_loss,
        )
        self.qem = model
        self.model_qem = model.prediction
        self.scs_qem = model.volume
        self.params_qem = model.params
        self.qem.voronoi_integration(plot=plot)
        self.scs_voronoi = self.qem.voronoi_volume

    def compare_scs_voronoi(self, folder_path=None, file_path=None, save=False):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        im = plt.scatter(
            to_numpy(self.qem.params["pos_x"]),
            to_numpy(self.qem.params["pos_y"]),
            s=1,
            c=self.scs_qem,
            cmap="viridis",
        )
        plt.gca().invert_yaxis()
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(r"QEM refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1, 3, 2)
        im = plt.scatter(
            to_numpy(self.qem.params["pos_x"]),
            to_numpy(self.qem.params["pos_y"]),
            s=1,
            c=self.scs_voronoi,
            cmap="viridis",
        )
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"Voronoi refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        im = plt.scatter(
            to_numpy(self.qem.params["pos_x"]),
            to_numpy(self.qem.params["pos_y"]),
            s=1,
            c=self.scs_voronoi - self.scs_qem,
            cmap="viridis",
        )
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"difference refined scs ($\AA^2$)")
        plt.clim(-self.scs_qem.mean() / 10, self.scs_qem.mean() / 10)
        plt.tight_layout()
        if save:
            if file_path is None:
                file_path = "voronoi_scs.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def compare_residual(
        self, mode="both", folder_path=None, file_path=None, save=False
    ):
        image = self.image
        if mode == "StatSTEM":
            plt.subplots(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("image")
            plt.subplot(1, 3, 2)
            im = plt.imshow(self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("StatSTEM prediction")
            plt.subplot(1, 3, 3)
            im = plt.imshow(image - self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 100, image.mean() / 100)
            plt.title("difference")
            plt.tight_layout()
        if mode == "QEM":
            plt.subplots(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("image")
            plt.subplot(1, 3, 2)
            im = plt.imshow(self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("QEM prediction")
            plt.subplot(1, 3, 3)
            im = plt.imshow(image - self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("difference")
            plt.tight_layout()
            plt.clim(-image.mean() / 100, image.mean() / 100)
        if mode == "both":
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Input Image")

            plt.subplot(2, 3, 2)
            im = plt.imshow(self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.tight_layout()
            plt.title("QEM Model")

            plt.subplot(2, 3, 3)
            diff = image - self.model_qem
            im = plt.imshow(diff)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 10, image.mean() / 10)
            plt.tight_layout()
            plt.title("Residuals")

            plt.subplot(2, 3, 4)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Input Image")

            plt.subplot(2, 3, 5)
            im = plt.imshow(self.model_statstem)
            plt.clim(image.min(), image.max())
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Legacy StatSTEM Model")

            plt.subplot(2, 3, 6)
            im = plt.imshow(image - self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 10, image.mean() / 10)
            plt.tight_layout()
            plt.title("Residuals")
        if save:
            if file_path is None:
                file_path = "residuals.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def compare_scs_map(self, folder_path=None, file_path=None, save=False):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        pos_x = to_numpy(self.params_qem["pos_x"]) * self.dx
        pos_y = to_numpy(self.params_qem["pos_y"]) * self.dx
        pos_x_statstem = self.output_coordinates[:, 0]
        pos_y_statstem = self.output_coordinates[:, 1]
        index_statstem_in_qem = np.array(
            [
                np.argmin(
                    np.sqrt((pos_x_statstem - x) ** 2 + (pos_y_statstem - y) ** 2)
                )
                for x, y in zip(pos_x, pos_y)
            ]
        )
        pos_x_statstem = pos_x_statstem[index_statstem_in_qem]
        pos_y_statstem = pos_y_statstem[index_statstem_in_qem]
        volume_statstem = volume_statstem[index_statstem_in_qem]

        plt.subplots(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        im = plt.scatter(pos_x, pos_y, c=volume_qem, s=2)
        # make aspect ratio equal
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"QEM refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1, 3, 2)
        im = plt.scatter(pos_x_statstem, pos_y_statstem, c=volume_statstem, s=2)
        plt.gca().invert_yaxis()
        plt.clim(volume_qem.min(), volume_qem.max())
        # make aspect ratio equal
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"Matlab StatSTEM refined scs ($\AA^2$)")
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        im = plt.scatter(pos_x, pos_y, c=volume_statstem - volume_qem, s=2)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"difference refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.clim(-volume_qem.mean() / 10, volume_qem.mean() / 10)
        if save:
            if file_path is None:
                file_path = "scs_map.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def compare_scs_histogram(self, folder_path=None, file_path=None, save=False):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        # remove outliners of volume_statstem
        data = volume_statstem.reshape(-1, 1)  # Reshape for compatibility with sklearn
        clf = IsolationForest(
            contamination=0.01
        )  # Estimate of the contamination of the data
        preds = clf.fit_predict(data)
        outliers = data[preds == -1].reshape(-1)
        # only keep the outliers that larger than the mean
        outliers = outliers[outliers > np.mean(volume_statstem)]
        # remove outliers from volume_statstem
        volume_statstem = volume_statstem[
            np.isin(volume_statstem, outliers, invert=True)
        ]
        volume_statstem = volume_statstem[volume_statstem > 0]  # remove negative values
        plt.figure(figsize=(6, 6))
        plt.hist(volume_qem, bins=100, alpha=0.5, label="QEM", density=True)
        plt.hist(volume_statstem, bins=100, alpha=0.5, label="StatSTEM", density=True)
        plt.xlabel(r"scs ($\AA^2$)")
        plt.ylabel("frequency")
        plt.legend()
        plt.title("Histogram of scs")
        if save:
            if file_path is None:
                file_path = "scs_histogram.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def scs_error(self, relative=True):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        if volume_qem.shape != volume_statstem.shape:
            pos_x = to_numpy(self.params_qem["pos_x"]) * self.dx
            pos_y = to_numpy(self.params_qem["pos_y"]) * self.dx
            pos_x_statstem = self.output_coordinates[:, 0]
            pos_y_statstem = self.output_coordinates[:, 1]
            index_statstem_in_qem = np.array(
                [
                    np.argmin(
                        np.sqrt((pos_x_statstem - x) ** 2 + (pos_y_statstem - y) ** 2)
                    )
                    for x, y in zip(pos_x, pos_y)
                ]
            )
            volume_statstem = volume_statstem[index_statstem_in_qem]
        mask = (volume_statstem > np.percentile(volume_statstem, 0.1)) & (
            volume_statstem < np.percentile(volume_statstem, 99.9)
        )
        if relative:
            error = (volume_statstem - volume_qem) / volume_qem
        else:
            error = volume_statstem - volume_qem
        return error[mask].mean(), error[mask].std()

    def position_error(self, units: str = "A") -> dict:
        """Per-atom position error vs StatSTEM (nearest-neighbour matched).

        Returns a dict with displacement statistics — mean, RMSE, p50,
        p95, and max — in the requested ``units`` (``"A"`` for Ångström
        i.e. ``self.dx``-scaled, ``"px"`` for pixel units). Plus the
        per-axis bias (mean Δx, mean Δy) which surfaces systematic
        offsets (e.g. half-pixel grid alignment differences).

        Matching: each QEM atom is paired with the nearest StatSTEM
        atom in Euclidean distance — same convention as
        :meth:`scs_error`. Atoms with no neighbour within ``2·dx``
        are dropped from the statistics (likely false detections on
        either side).
        """
        scale = float(self.dx) if units.lower().startswith("a") else 1.0
        pos_x = to_numpy(self.params_qem["pos_x"]) * scale
        pos_y = to_numpy(self.params_qem["pos_y"]) * scale
        pos_x_ref = self.output_coordinates[:, 0]
        pos_y_ref = self.output_coordinates[:, 1]
        if not units.lower().startswith("a"):
            pos_x_ref = pos_x_ref / float(self.dx)
            pos_y_ref = pos_y_ref / float(self.dx)

        # Nearest-neighbour match (QEM → StatSTEM). Vectorised: build
        # a pairwise distance matrix, take argmin per row.
        dx = pos_x[:, None] - pos_x_ref[None, :]
        dy = pos_y[:, None] - pos_y_ref[None, :]
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2, axis=1)
        d_min = np.sqrt(d2[np.arange(d2.shape[0]), idx])
        delta_x = pos_x - pos_x_ref[idx]
        delta_y = pos_y - pos_y_ref[idx]

        # Drop unmatched atoms (>2·dx in user units, or 2 px in px units).
        cutoff = 2.0 * (float(self.dx) if units.lower().startswith("a") else 1.0)
        keep = d_min < cutoff
        if not keep.any():
            return {"n_matched": 0, "units": units}

        d = d_min[keep]
        return {
            "n_matched": int(keep.sum()),
            "n_total": int(d_min.size),
            "units": "A" if units.lower().startswith("a") else "px",
            "mean": float(d.mean()),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "p50": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
            "bias_x": float(delta_x[keep].mean()),
            "bias_y": float(delta_y[keep].mean()),
        }

    def report(self) -> dict:
        """One-shot summary: residual + position + SCS error vs StatSTEM.

        Prints a compact report and returns the underlying dict so it
        can be tabulated across benchmark samples.
        """
        # Image-residual stats
        res_qem = self.image - self.model_qem
        res_st = self.image - self.model_statstem if hasattr(self, "model_statstem") else None
        out: dict = {
            "residual_std_qem": float(np.std(res_qem)),
            "residual_l1_qem": float(np.mean(np.abs(res_qem))),
        }
        if res_st is not None:
            out["residual_std_statstem"] = float(np.std(res_st))
            out["residual_l1_statstem"] = float(np.mean(np.abs(res_st)))

        # Position metrics in Å
        try:
            pos = self.position_error(units="A")
            out["position"] = pos
        except Exception as exc:  # pragma: no cover
            out["position_error"] = str(exc)

        # SCS comparison
        try:
            mean, std = self.scs_error(relative=True)
            out["scs_rel_error_mean"] = float(mean)
            out["scs_rel_error_std"] = float(std)
        except Exception as exc:  # pragma: no cover
            out["scs_error"] = str(exc)

        # Pretty-print
        print(f"  residual std (image)  QEM: {out['residual_std_qem']:.4f}",
              end="")
        if "residual_std_statstem" in out:
            print(f"   StatSTEM: {out['residual_std_statstem']:.4f}", end="")
        print()
        if "position" in out and out["position"].get("n_matched", 0) > 0:
            p = out["position"]
            print(f"  position vs StatSTEM  mean={p['mean']:.4f} {p['units']}"
                  f"   rmse={p['rmse']:.4f}   p95={p['p95']:.4f}"
                  f"   max={p['max']:.4f}   (n={p['n_matched']}/{p['n_total']})")
            print(f"  position bias         Δx={p['bias_x']:+.4f}"
                  f"   Δy={p['bias_y']:+.4f} {p['units']}")
        if "scs_rel_error_mean" in out:
            print(f"  SCS rel error         mean={out['scs_rel_error_mean']:+.4f}"
                  f"   std={out['scs_rel_error_std']:.4f}")
        return out
