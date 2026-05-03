# Optics Stack Refactor — Design (2026-05-03)

> Replace `qem/instruments/probe.py` + `qem/instruments/ctf.py` with a small
> set of focused modules under `qem/instruments/optics/`, and clean up the
> CTF surface used by `qem/fit/`.  Breaking changes are allowed; a thin
> compatibility shim covers the previous public API for one release.

## Goals

1. **Clarity.**  Every file does one thing.  Physics primitives, data
   containers, and high-level "build a PSF" operations are separated.
2. **Single source of truth.**  One sign convention (abtem's
   `defocus = -C10`).  One FFT convention (q-space fftshift-centered).
   One way to specify probe parameters.
3. **Performance.**  PyTorch backend so PSFs compose with the existing
   PyTorch fit loops without numpy↔torch round-trips, and so they run on
   GPU when the fit does.
4. **Numerical equivalence with abtem.**  Already verified for `chi`,
   temporal and spatial envelopes; preserved by construction in the
   refactor.

## Non-goals

- Re-architecting `qem/fit/image_fitting.py` or `convolve_fitting.py`
  beyond what's needed to consume the new optics surface.
- Multislice / wave propagation (`focused_probe`, `plane_wave_illumination`,
  Cc averaging) — these stay as-is in a `qem/instruments/wave.py`
  carve-out for now and can be refactored separately later.
- Detector models (`qem/instruments/detector.py`) — unchanged.

## Current state — what's wrong

| Symptom | Underlying cause |
|---|---|
| `class Aberration(Aberration)` self-rebinding | Two ctors-via-arg-count hack instead of one canonical ctor + named factory. |
| Per-instance `Krivanek`, `Haider`, `Description` strings | Class metadata stored as instance state. |
| `df` parameter and C10 aberration use opposite sign | Two competing entry points for the same physics. |
| `_chi_gradients` duplicates the polynomial expansion in `chi()` | No shared low-level primitive. |
| 4 CTF subclasses each override `get_psf` | Base class FFT convention only round-trips for symmetric CTFs. |
| 4 ways to specify probe params (`Probe`, `ProbeParameters`, `create_probe_parameters`, raw kwargs) | Iterative growth without consolidation. |
| `ADF_CTF` produced empty PSFs (just fixed) | `|probe(q)|²` masked by detector annulus has empty intersection with aperture support. |
| `np.product` deprecation, `numpy.fft` and `scipy.fft` both used | No unified backend. |
| Pyright errors throughout | Optional types tossed into positional args. |

## Target architecture

```
qem/instruments/
  __init__.py                # re-export public API
  optics/
    __init__.py              # public API for optics
    constants.py             # wavev, relativistic_mass_correction, mrad↔invÅ
    aberrations.py           # Aberrations dataclass (polar coefficients)
    chi.py                   # chi() and grad_chi() — pure tensor functions
    envelopes.py             # temporal_envelope(), spatial_envelope(), focal_spread_from_chromatic()
    aperture.py              # hard_aperture(), soft_aperture()
    grid.py                  # Grid dataclass + alpha_phi_from_grid() helper
    probe.py                 # Probe dataclass + probe_wave() function
    psf.py                   # ssb_psf(), adf_psf(), icom_psf(), epie_psf()
  detector.py                # unchanged
  tilt.py                    # unchanged
  wave.py                    # focused_probe(), plane_wave_illumination(), Cc averaging (carve-out)
  _legacy.py                 # compatibility shim for old public names (deprecated)
```

Approximate sizes: each `optics/*.py` ≤ 200 lines, total ≤ 1000 lines
(down from ~1940).  The legacy shim adds ~150 lines that will be deleted
in the next major release.

### Module responsibilities

**`constants.py`** — pure functions:
- `wavev(energy_eV) -> float` (Å⁻¹), unchanged formula.
- `wavelength(energy_eV) -> float` (Å), `1/wavev`.
- `relativistic_mass_correction(energy_eV) -> float`.
- `mrad_to_invA(mrad, energy_eV) -> float` and back.

**`aberrations.py`** — single `Aberrations` dataclass with named polar
coefficient fields matching abtem's symbol set
(`C10, C12, phi12, C21, phi21, C23, phi23, C30, C32, phi32, …, C56, phi56`).
Aliases as properties (`defocus = -C10`, `Cs = C30`, …).
Convenience constructor `Aberrations.from_dict({...})`.
No per-instance string metadata; `__repr__` and `__str__` look up symbols
from a module-level table.

**`chi.py`** — two functions, both pytorch-tensor in/out:
- `chi(alpha, phi, *, wavelength, aberrations) -> Tensor` — phase, radians.
- `grad_chi(alpha, phi, *, wavelength, aberrations) -> tuple[Tensor, Tensor]`
  — `(∂χ/∂k, ∂χ/∂φ)`, used by the spatial envelope.

Both share a single Horner-style polynomial evaluator over α, indexed by
the polar `Aberrations` fields.  No defocus-vs-list duality:
`Aberrations.defocus` is the only user-facing name; internally it sets
`C10 = -defocus`.

**`envelopes.py`** — pure functions on tensors:
- `temporal_envelope(alpha, *, wavelength, focal_spread) -> Tensor`
- `spatial_envelope(alpha, phi, *, wavelength, aberrations, angular_spread_mrad) -> Tensor`
- `focal_spread_from_chromatic(Cc_A, dE_eV, energy_eV, *, convention='1/e') -> float`

These match abtem exactly (verified in the prior commit).

**`aperture.py`** — `hard_aperture(alpha, semi_angle_mrad)` and
`soft_aperture(alpha, phi, semi_angle_mrad, angular_sampling)` mirroring
abtem's helpers.  Used by both `probe_wave()` and the Hofer SSB formula.

**`grid.py`** — `Grid` dataclass `(pix: tuple[int,int], real: tuple[float,float])`
with helpers:
- `Grid.q_array() -> Tensor[2, ny, nx]` (fft-natural, in Å⁻¹).
- `Grid.alpha_phi(wavelength) -> tuple[Tensor, Tensor]` (radians).
- `Grid.dx, Grid.dy, Grid.dq_x, Grid.dq_y` properties.

Always fft-natural inside (DC at `[0, 0]`); centering is the caller's
responsibility via `torch.fft.fftshift`.

**`probe.py`** — `Probe` dataclass:
```python
@dataclass(frozen=True)
class Probe:
    energy: float                       # eV
    aperture: float                     # mrad
    aberrations: Aberrations = ()       # empty Aberrations()
    focal_spread: float | None = None   # Å, 1/e half-width (Kirkland convention)
    angular_spread: float | None = None # mrad (source angular size)
    soft_aperture: bool = True
```
Plus `probe_wave(grid, probe, *, device, dtype) -> Tensor` returning the
focused probe wave function in real space, fftshift-centered.

**`psf.py`** — four primary PSF functions plus their q-space companions:
- `ssb_psf(grid, probe) -> Tensor` / `ssb_ctf(grid, probe) -> Tensor`
- `adf_psf(grid, probe) -> Tensor` / `adf_ctf(grid, probe) -> Tensor`
- `icom_psf(grid, probe, *, high_pass_mrad=None) -> Tensor` / `icom_ctf(...)`
- `epie_psf(grid, probe) -> Tensor` / `epie_ctf(grid, probe) -> Tensor`

`*_psf` return real-space PSFs, fftshift-centered, real `Tensor`.
`*_ctf` return q-space transfer functions, fft-natural ordering, complex
`Tensor`.  For SSB the natural form IS the CTF (analytical Hofer
formula); the PSF wraps an IFFT.  For ADF the natural form is the PSF
(`|probe(r)|²`); the CTF wraps an FFT.  Each pair shares the heavy
computation through an internal helper so calling both costs only one
extra FFT.

### Sign and FFT conventions (single source of truth)

- **Defocus / C10:** `Aberrations.defocus = -Aberrations.C10` (abtem
  convention).  `defocus > 0` ⇒ under-focus.  Documented in
  `aberrations.py` and at the top of the design doc.
- **chi sign in wave function:** `ψ(k) = aperture(k) · exp(-i χ(k))`
  (Kirkland Eq. 5.16, abtem `complex_exponential(-array)`).
- **q-space arrays:** fft-natural unless caller `fftshift`s.  PSFs from
  `psf.py` are returned fftshift-centered.  All round-trips use
  `fftshift(fft2(ifftshift(x)))` so they are correct for non-symmetric x.

### Backend & device

- `Probe` and `Aberrations` are plain Python dataclasses (not tensors).
- All physics functions accept and return `torch.Tensor`.  Inputs are
  promoted via `torch.as_tensor`; numpy users call
  `.cpu().numpy()` on results.
- Functions take optional `device` and `dtype` kwargs (default
  `torch.get_default_dtype()` and CPU).
- `psf.py` exposes `numpy=True` shortcut that returns `np.ndarray`
  (single round-trip) for callers that don't want torch.

### Migration & compatibility

`qem/instruments/_legacy.py` re-exports the old public names with
`DeprecationWarning`s and adapts the new API:

| Old name | New equivalent |
|---|---|
| `Probe(eV=…, aperture=…, df=…, …)` | `Probe(energy=…, aperture=…, aberrations=Aberrations(defocus=…), …)` |
| `Aberration("C12", "A1", "...", amp, ang, n, m)` | `Aberrations(C12=amp, phi12=ang)` |
| `aberration_starter_pack()` | `Aberrations()` (all zeros, all symbols available) |
| `chi(q, qphi, lam, df, aberrations)` | `chi(alpha=q*lam, phi=qphi, wavelength=lam, aberrations=Aberrations.from_old_list(aberrations, df))` |
| `SSB_CTF(...).get_psf(pix, real)` | `ssb_psf(Grid(pix, real), Probe(...))` |
| `ADF_CTF(...).get_psf(pix, real)` | `adf_psf(Grid(pix, real), Probe(...))` |
| `ePIE_CTF(...).get_psf(pix, real)` | `epie_psf(...)` |
| `iCoM_CTF(...).get_psf(pix, real)` | `icom_psf(...)` |
| `ProbeParameters` / `create_probe_parameters` | `Probe(...)` directly |
| `create_aberration_list(defocus=…, Cs=…)` | `Aberrations(defocus=…, Cs=…)` |
| `make_contrast_transfer_function(...)` | `probe_wave(Grid(...), Probe(...))` returns the real-space probe; FFT it for the q-space CTF. |

Internal `qem/fit/` consumers (`convolve_fitting.py`,
`ptychography_optimization.py`, `point_potential.py`) are migrated to
the new API as part of this work — they're not external users.  Tests
in `tests/test_ctf.py` are updated to the new API; old behaviour stays
covered through the shim until the next major version.

`focused_probe`, `plane_wave_illumination`, `simulation_result_with_Cc`,
`Cc_integration_points`, `Cc_defocus_spread`, `convert_deltaE` move to
`qem/instruments/wave.py` essentially as-is.  They're not
restructured — that's a separate task.

### Helpers leaving the optics module

- `demonstrate_aberration_effects` → `examples/aberration_effects.py`
  (it's a plotting demo, not a library function).
- `extract_psf_from_atom_image` → `qem/processing/psf.py` (image analysis).
- `calculate_psf_width` → `qem/processing/psf.py` (image analysis).

## Testing strategy

- Numerical equivalence tests vs. abtem-derived reference values
  (already exist as `/tmp/compare_qem_abtem.py`; promote into
  `tests/test_optics_vs_abtem.py`).  Cover: `chi` for defocus, Cs,
  astigmatism, coma; temporal envelope; spatial envelope.
- Round-trip tests: `psf = ssb_psf(grid, probe)`, then
  `ifft2(fft2(psf))` returns `psf` to within fp tolerance.
- API parity tests: every old `tests/test_ctf.py` case has a new-API
  equivalent and an old-API equivalent (latter exercises the shim).
- Performance microbenchmark in `qem/benchmarks/` comparing old SSB on
  CPU numpy vs. new on CPU torch vs. new on CUDA torch.
- Pyright clean: fix all reportArgumentType issues in the touched files.

## Risks

- **Behaviour change in `qem/fit/`**: the SSB/ADF defocus sign change
  could shift fit results.  Mitigation: pin two regression tests under
  `tests/test_fit_regression.py`, each running an existing
  `convolve_fitting` example end-to-end and asserting that fitted
  positions match the saved baseline to ≤ 0.01 Å and fitted phases /
  intensities to ≤ 1e-4 relative.  Baselines are regenerated once
  using the *new* code (post-refactor) and saved as `.npz`; the test
  guards against future regressions, not against the deliberate
  defocus-sign change.
- **PyTorch float32 default precision**: SSB/ADF analytical formulas
  involve `arccos` near boundaries that can be touchy in fp32.
  Mitigation: keep the inner `arccos`/`sqrt` block in fp64 and cast
  back; or expose a `dtype=torch.float64` knob.
- **GPU presence not guaranteed in CI**: keep CPU-only as the default
  path; gate CUDA tests on `torch.cuda.is_available()`.

## Acceptance criteria

1. `tests/` passes (133+ tests today, plus the new ones).
2. `tests/test_optics_vs_abtem.py` matches abtem to ≤ 1e-9 relative on
   chi, ≤ 1e-12 on the envelopes.
3. `examples/partial_coherence_demo.py` and
   `examples/aberration_effects_on_ctf.py` regenerate visually
   identical figures (or better — the spatial envelope already shows
   non-trivial damping after the prior commit).
4. `pyright qem/instruments/optics/` reports zero errors.
5. No file in `qem/instruments/optics/` exceeds 250 lines.
6. SSB PSF on CPU is no slower than the current numpy implementation;
   on CUDA it's at least 5× faster for ≥ 256² grids.

## Out-of-scope (future work)

- Refactor `qem/instruments/wave.py` (the carve-out) — multislice
  helpers deserve their own pass.
- Refactor `qem/fit/image_fitting.py` (3372 lines).
- Replace the `qem.utils.torch_compat as keras` shim with direct torch
  imports across `qem/fit/`.
