"""Grid: pixel/real-space sampling pair, plus q/α/φ helpers.

A :class:`Grid` is just ``(pixels, extent)``.  It builds the q-space grid
(in Å⁻¹, fft-natural ordering — DC at ``[0, 0]``) and the angular grid
``(α, φ)`` (in radians), as PyTorch tensors on a chosen device/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    """Sampling parameters for a 2D image.

    Parameters
    ----------
    pixels : (ny, nx)
        Pixel dimensions.
    extent : (Ly, Lx)
        Real-space extent in Å.
    """

    pixels: tuple[int, int]
    extent: tuple[float, float]

    @property
    def ny(self) -> int:
        return int(self.pixels[0])

    @property
    def nx(self) -> int:
        return int(self.pixels[1])

    @property
    def Ly(self) -> float:
        return float(self.extent[0])

    @property
    def Lx(self) -> float:
        return float(self.extent[1])

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dq_y(self) -> float:
        return 1.0 / self.Ly

    @property
    def dq_x(self) -> float:
        return 1.0 / self.Lx

    @property
    def angular_sampling_mrad(self) -> tuple[float, float]:
        """Angular sampling Δα = λ · Δq.  Requires a wavelength to compute."""
        raise AttributeError(
            "angular_sampling_mrad needs a wavelength; "
            "use angular_sampling(wavelength) instead."
        )

    def angular_sampling(self, wavelength_A: float) -> tuple[float, float]:
        """(Δα_y, Δα_x) in mrad given the electron wavelength."""
        return (1e3 * wavelength_A * self.dq_y,
                1e3 * wavelength_A * self.dq_x)

    def q_grid(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reciprocal-space grid (qy, qx) in Å⁻¹, fft-natural ordering.

        Both tensors have shape ``(ny, nx)`` after broadcasting via
        ``torch.meshgrid`` with ``indexing='ij'``.
        """
        dtype = dtype or torch.get_default_dtype()
        qy = torch.fft.fftfreq(self.ny, d=self.dy, device=device, dtype=dtype)
        qx = torch.fft.fftfreq(self.nx, d=self.dx, device=device, dtype=dtype)
        qy_g, qx_g = torch.meshgrid(qy, qx, indexing="ij")
        return qy_g, qx_g

    def alpha_phi(
        self,
        wavelength_A: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Angular grid (α, φ) in radians for a given electron wavelength.

        ``α = λ · |q|`` is the polar semi-angle and
        ``φ = atan2(qy, qx)`` is the azimuth.  Both have shape
        ``(ny, nx)`` and ordering matches :meth:`q_grid` (fft-natural).
        """
        qy, qx = self.q_grid(device=device, dtype=dtype)
        alpha = wavelength_A * torch.sqrt(qy * qy + qx * qx)
        phi = torch.atan2(qy, qx)
        return alpha, phi
