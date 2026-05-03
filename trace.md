# Trace: qem

<!-- concepts: backend-abstraction, packaging, error-handling -->

## 2026-05-01 — HYP-21: fix critical issues from HYP-20 review

Working on Multica issue HYP-21 on branch `fix/qem-critical-issues-hyp21` (forked from `feat/sto_interface`). Pre-existing uncommitted changes on the parent branch are kept untouched on this branch.

Plan, by priority:

- **P1** Backend-agnostic CUDA cleanup. `qem/fit/linear_solver.py:702` calls `torch.cuda.empty_cache()` even when `torch` was never imported in the local scope (the `if fit_background:` branch is the only place that imports it), causing `UnboundLocalError` when `fit_background=False`. Same pattern repeats inside `_create_sparse_matrix`. `qem/fit/image_fitting.py:1865-1892` also calls `empty_cache()` from generic Keras code paths.
  - Approach: add `release_backend_memory()` helper in `qem/utils/backend.py` that no-ops unless the active Keras backend is torch *and* CUDA is actually available. Replace inline `import torch; torch.cuda.empty_cache()` calls with it.
- **P2** Packaging. Add `pyproject.toml` (PEP 621), Python >= 3.10, split heavy deps into extras (`gui`, `docs`, `dev`, `io`, backend-specific). Drop the `git+https://...matplotlib-scalebar` from required deps. Deprecate `setup.py` to a thin shim.
- **P2** Import boundaries. `qem/__init__.py` runs `from .app import *`, which imports Streamlit and calls `st.set_page_config()` at import time, breaking headless usage. Drop the import; expose the Streamlit app via a console entry point (`qem-app`).
- **P3** Error handling. `ImageFitting.linear_estimator` catches every `Exception`, logs, and silently returns the input params. Make the broad `except` opt-in via a `best_effort` flag; default to re-raising domain-specific errors so failures aren't hidden.

Reproduction baseline: `KERAS_BACKEND=torch pytest tests/test_linear_solver.py::TestIntegration::test_full_linear_estimation_workflow -x` -> `UnboundLocalError` at `linear_solver.py:702`. Will rerun after each fix.

### EARS — Progress (2026-05-01 16:32)
<!-- concepts: backend-abstraction, error-handling -->

P1 in progress. Added `release_backend_memory()` helper in `qem/utils/backend.py` — gates cleanup on `keras.backend.backend() == "torch"` *and* `torch.cuda.is_available()`. Replaced four bare `torch.cuda.empty_cache()` call sites in `qem/fit/linear_solver.py` (build_sparse_matrix bg branch, build_sparse_matrix tail, _create_sparse_matrix MPS branch) and wired the import into `qem/fit/image_fitting.py` ahead of replacing its two GPU-cache calls in stochastic fitting.

Decision: keep `release_backend_memory()` in `qem/utils/backend.py` rather than a new module, since `backend.py` already owns backend detection (`detect_available_backends`, `configure_backend`). Future MPS-cache support (`torch.mps.empty_cache()`, available in newer PyTorch) can be added in the same helper without touching call sites.

Pyright "module not found" diagnostics on every edit are environment noise — Pyright in the editor isn't pointed at the conda env, but `KERAS_BACKEND=torch python -c "import keras, torch"` works fine. Ignoring.

Next: finish replacing the two image_fitting.py call sites, then move on to packaging (P2).

### EARS — Progress (2026-05-01 17:19)
<!-- concepts: backend-abstraction, packaging -->

HYP-23: fixing core import failure flagged by Codex review of HYP-21. Confirmed bug: `KERAS_BACKEND=tensorflow python -c "import qem"` raises `ModuleNotFoundError: No module named 'tensorflow'` because `keras>=3.0` is core but backends are extras-only and Keras 3 defaults to TF.

Decision: do the auto-detect in `qem/__init__.py` (option 3 from Codex's list) instead of lazy-loading `qem.fit` (option 2). Auto-detect is invisible to callers and preserves the eager-import API; lazy-loading would break code that does `qem.ImageFitting`/`qem.fit`. Picked preference order torch > jax > tensorflow (matches `qem.utils.backend.get_best_backend`), with `numpy` as the always-importable fallback so `import qem` never fails. User-set `KERAS_BACKEND` is respected via early return.

Inlining the detection in `qem/__init__.py` rather than calling `qem.utils.backend.auto_configure()` to avoid pulling `qem.utils.*` into the import-time path before the package's own subpackage layout is established.

### EARS — Progress (2026-05-01 23:38)
<!-- concepts: pytorch-migration, code-review-cleanup, backend-pruning -->
HYP-27 review follow-up: closing the gap on PyTorch migration with breaking-change allowance ("super clean"). User OK'd dropping back-compat shims, so I'm pruning legacy TF/JAX paths instead of keeping multi-backend dispatch dead.

Started in `qem/fit/linear_solver.py`: removed `BackendSolver` `Protocol`-only stragglers I didn't need, dropped unused imports, simplified `MemoryInfo` (no `backend` field — only torch left). Still need to delete `TensorFlowSolver`, `JAXSolver`, the TF/JAX branches in `_create_sparse_matrix`, and the multi-backend `SOLVERS`/`MEMORY_THRESHOLDS` maps. Same plan for `qem/fit/model.py` (`if backend == 'jax'` / `'tensorflow'` branches) and the TF cleanup in `qem/utils/memory_optimization.py`. The Pyright import diagnostics on `numpy` / `qem.*` are environment noise (this trace lives outside the project's pyrightconfig); they don't reflect actual missing modules.

Pending: validate_params chained-comparison bug (`pos_x_len != pos_y_len != height_len`), `quickstart.rst` 5-vs-6 args for `gaussian_2d_single`, `torch_compat._dtype` not handling Python `bool`, `tests/test_precision_config.py` returning bools from tests, and the `photutils` warning at `qem` import time.

### EARS — Progress (2026-05-02 11:02)
<!-- concepts: pytorch-migration, backend-abstraction -->

User hit `TypeError: unsqueeze(): argument 'input' (position 1) must be Tensor, not numpy.ndarray` running `model.fit_stochastic(...)` in the introduction notebook. Root cause: `qem.utils.torch_compat._Ops.__getattr__` aliases `expand_dims` straight to `torch.unsqueeze` (no `_as_tensor` wrap), and `update_from_local_params` was passing `np.where(mask)[0]` directly. `_Ops.scatter_update` already calls `.long()` on indices but never gets a chance because `expand_dims` blows up first.

Fix scope chosen: wrap at the two known-numpy call sites (`update_from_local_params` line 1931 and the sibling `batch_indices` path around line 1562) rather than fixing the shim. Reason: `_as_tensor` always calls `.cpu()` (torch_compat.py:61), so promoting `expand_dims` to wrap inputs would silently force GPU→CPU transfers across all the existing tensor-input call sites in `linear_solver.py`/`model.py`/`image_fitting.py`. That's a separate problem — flagged but not fixed here. Used `dtype="int64"` explicitly since torch scatter wants int64.

Sibling line 1562 (`batch_indices` from `np.random.permutation`) has the same latent bug; only triggers when `batch_size < num_coordinates`. Patching both.

Pyright's new h5py/Datatype diagnostics on `image_fitting.py` (lines 260, 323, 1073, 1990, etc.) are pre-existing environment noise — not introduced by this edit.

### EARS — Progress (2026-05-02 14:14)
<!-- concepts: pytorch-migration, examples-validation, public-api-surface -->
HYP-27 follow-up: validating that examples/ and tutorials still run after the migration cleanup. Survey turned up two real breakages and one accidental-API surface gap:

1. `examples/aberration_examples.py` imports `SSB_CTF`, `ADF_CTF`, `create_aberration_list`, `aberration_starter_pack`, `demonstrate_aberration_effects` from the top-level `qem.instruments` package, plus `PtychographyOptimizer` from `qem.fit`. Both packages had stale `__init__.py` re-export lists (e.g. `qem/instruments/__init__.py` was advertising `aberration_function` / `contrast_transfer_function` symbols that don't exist anywhere). Decided to fix this at the package boundary — `qem.instruments` and `qem.fit` should explicitly re-export the user-facing API, not silently rely on `from .x import *`.
2. `examples/aberration_effects_on_ctf.py` was instantiating `aberration(...)` (lowercase legacy class) without importing it. Renamed those call sites to use the existing `Aberration` compat wrapper, which already accepts the 7-arg legacy form.
3. `examples/ETO_KTO/{100,110}/ETO_KTO.py` use the dropped `from qem.image_fitting import ImageModelFitting`, plus a `%matplotlib qt` cell-magic line that makes the file invalid Python outside of Jupyter. These scripts are also pinned to absolute paths under `/home/zzhang/OneDrive/...` that don't exist in this checkout. They are research notebooks-as-`.py` rather than runnable demos. Plan: at minimum fix the import line so they don't fail at parse time for someone with the data; flag the cell-magic + missing data as known caveats in the status comment.

Notebooks (`.ipynb`) and `.rst` doc snippets still need a smoke pass — focus next on import-time compatibility, not full execution, since they probably need data files too.

### EARS — Progress (2026-05-02 14:25)
<!-- concepts: examples-validation, public-api-surface, ptychography-autograd, scalebar-units -->
Continuing the HYP-27 examples sweep. New findings since the previous trace:

- `qem.fit.PtychographyOptimizer` was migrated to call `model.fit(...)` for autograd, but `ConvolutionImageModel.call` was converting params to numpy and running scipy `fftconvolve` mid-forward — that breaks the autograd graph in any backend, not just torch (Keras/JAX/TF would have hit the same wall). Replaced the forward path with a fully-differentiable bilinear-scatter point-potential map plus FFT convolution implemented in pure PyTorch (`_bilinear_point_potential` + `_fft_convolve_same` helpers in `qem/fit/ptychography_optimization.py`). Example 5 in `aberration_examples.py` now actually runs the gradient loop.
- `torch_compat.Model.add_weight` was missing the `trainable` kwarg, and `torch_compat.ops.maximum` failed on `(tensor, scalar)` — both are Keras-isms the migrated code still relies on. Added `trainable` (mapped to `requires_grad`) and made `maximum`/`minimum` coerce scalars via `_as_tensor`.
- `qem/instruments/__init__.py` and `qem/fit/__init__.py` had stale `__all__` lists that didn't match the actual public API. Rewrote both as explicit re-export blocks (CTF helpers, `Probe`, `Aberration`, `SampleTilt`, `PtychographyOptimizer`, `ADFConvolutionFitting`, etc.) so the documented `from qem.instruments import …` / `from qem.fit import …` lines in examples and the ptychography markdown actually resolve.
- `examples/ETO_KTO.ipynb`, `examples/dev.ipynb`, and the two `examples/ETO_KTO/{100,110}/ETO_KTO.py` scripts referenced the dropped `qem.image_fitting` / `qem.model` paths and the old `ImageModelFitting` class name. Patched all of them to the new public API; also commented out `%matplotlib qt` so the `.py` files at least parse outside Jupyter. Note: those four files are gitignored — fixes are local only and won't ship in a PR.
- `dev.ipynb` cell 0 was using a long-gone `ImageModel.sum(X=, Y=, height=…)` keyword API. Rewrote the cell to use the current `set_params → build → sum(X, Y)` flow plus `safe_convert_to_numpy` so `plt.imshow` doesn't trip on a grad-requiring tensor.
- `docs/source/tutorials/basic_analysis.rst` still pointed at `qem.processing.butterworth_filter` and `qem.select.InteractivePlot`, neither of which exist. Switched to `qem.processing.signal.butterworth_window` and `qem.visualization.select.InteractivePlot`.
- `ImageFitting.scalebar` blew up with `ValueError: Invalid unit (A) with dimension` because matplotlib-scalebar only accepts SI length units. Mapped Å → metres internally so the legacy `units='A'` path keeps working.

Pending: rerun pytest after the scalebar/instruments changes, smoke-run `introduction.ipynb` end-to-end with the local Au data file, then post the status comment.

### EARS — Session Start (2026-05-03 22:01)
<!-- concepts: branch-wrap-up, stash-merging, git-workflow -->
- Task: Commit current changes on feat/pytorch-migration, merge stash f649c14441, then merge to master.
- Why: User wants to consolidate in-flight pytorch-migration work (current dirty tree + stashed WIP) and integrate it into the main branch.

### EARS — Progress (2026-05-03 22:15)
<!-- concepts: stash-merging, conflict-resolution, public-api-surface -->
Applied stash f649c14 onto fresh feat/pytorch-migration commit. Four files conflicted; resolutions:
- `qem/instruments/probe.py`: stash had a parallel rewrite (different big Probe class, removed `chi` / `make_contrast_transfer_function`, renamed `class aberration` → `class Aberration`). Restored entirely from HEAD — upstream design (small Probe + free CTF helpers) supersedes.
- `examples/introduction.ipynb`: 11 conflict blocks in cell JSON. Restored from HEAD — too risky to merge manually, upstream version reflects the migrated state.
- `qem/fit/__init__.py`: kept upstream's explicit-import structure but added the stash's still-valid extra re-exports (point_potential helpers, `OptimizationResult`, `fit_ssb_ptychography`, `fit_adf_image`, `GaussianKernel`, `voronoi_point_record`).
- `qem/instruments/__init__.py`: kept upstream's structure and added stash's valid extras (`detector` module, `chi`/`make_contrast_transfer_function`/`wavev` re-exports, `tilt_from_affine`).
- `image_fitting.py` and `model.py` auto-merged cleanly — no manual work.

### EARS — Progress (2026-05-03 22:46)
<!-- concepts: ctf, aberrations, partial-coherence -->
Debugging qem/instruments CTF/aberration/envelope code against ~/code/abtem reference.
Numerical comparison (test script /tmp/compare_qem_abtem.py) confirmed three concrete bugs:

1. **Defocus sign convention** — qem `df=+50` produces chi opposite sign of abtem
   `defocus=+50` (abtem stores `defocus = -C10`). qem currently matches abtem's
   internal `C10` sign, but the user-facing `df` parameter has the opposite
   sign of abtem's `defocus` property. Likely the visible "aberration looks wrong"
   complaint.

2. **Temporal envelope** off by factor of 2 in exponent. qem:
   `exp(-(πλΔk²)²/2)`; abtem (Kirkland 1/e half-width form):
   `exp(-(πλΔk²)²/4)`. At α=30 mrad, qem→0.22 vs abtem→0.47 — qem decays √2× too
   fast. Same numerical input.

3. **Spatial envelope** is fundamentally wrong. qem applies a flat
   `exp(-(σq)²/2)` Gaussian that ignores aberrations entirely and is
   dimensionally inconsistent (treats `source_size` in mrad as if it were Å).
   Correct quasi-coherent formula uses `|∇χ|²` and is exactly 1 for an
   unaberrated probe; qem currently damps even a pure-aperture probe.

Fix in progress in `qem/instruments/probe.py`:
- Negate `df` in `chi()` so `df` matches abtem's `defocus` sign
- Replace temporal envelope with `exp(-((πλΔk²)/2)²)`
- Replace spatial envelope with `|∇χ|²` formula (needs a `_chi_gradients`
  helper and `qphi` to be plumbed through `CTF.get_psf`)
- Update `create_aberration_list(defocus=X)` to map to `Aberration(C10, -X)`

Affected tests: `test_larger_source_size_gives_narrower_envelope` no longer
makes sense for unaberrated probes — must be updated to include defocus or
Cs.

### EARS — Progress (2026-05-03 23:03)
<!-- concepts: ctf, adf-imaging, partial-coherence, demos -->
Followup from the abtem-alignment fix in qem/instruments. User reported
two new issues with examples/partial_coherence_demo.py:

1. **Spatial envelope flat at 1**: After the `|∇χ|²` fix, an unaberrated
   probe correctly gives envelope = 1 (matches abtem). The demo creates
   probes with no `df`/aberrations, so the spatial curve is identically 1.
   This is correct physics but uninformative. Need to update the demo to
   add realistic defocus / Cs so the spatial envelope shows non-trivial
   damping.

2. **ADF PSF blank**: `ADF_CTF.calculate_ctf` had a fundamental design bug:
   it computed |probe(q)|² (only nonzero inside aperture, q ≤ 20 mrad)
   then masked by detector geometry (q ≥ 50 mrad). Empty intersection →
   identically zero. Fixed by replacing with the standard incoherent ADF
   PSF: |probe(r)|² in real space (Z-contrast, thin specimen). Detector
   inner/outer angles are kept for API compat but only affect overall
   signal scaling, not PSF shape.

### EARS — Progress (2026-05-03 23:13)
<!-- concepts: ctf, adf-imaging, fft-conventions -->
Wrapping up the abtem-alignment fixes. Final state:
- ADF_CTF rewritten to use the standard incoherent ADF model
  (PSF = |probe(r)|² in real space). Old code masked |probe(q)|² inside the
  aperture by the detector annulus, giving identically zero PSFs whenever
  aperture < detector_inner.
- ADF_CTF.get_psf overridden with clean FFT conventions; the base-class
  ifft2(ifftshift(...)) round-trip only happens to work for symmetric CTFs.
- partial_coherence_demo.py now uses representative uncorrected STEM
  (defocus = 50 Å, Cs = 1 mm) so the spatial envelope shows non-trivial
  damping. With ∇χ = 0 the abtem-style envelope is exactly 1, which is
  correct physics but uninformative without aberrations.
- Cleaning up duplicate `Aberration` import that the linter introduced in
  qem/instruments/__init__.py when it auto-renamed `class aberration` →
  `class Aberration`.

### EARS — Progress (2026-05-03 23:24)
<!-- concepts: ctf, refactor, optics-architecture -->
Brainstorming a from-first-principles refactor of qem/instruments/{probe,ctf}.py.
User asked for high clarity + high performance, breaking changes OK, drop tech debt.

Scoped at "Option B" (instruments/ + CTF surface in qem/fit/). Deferred broader
refactor of image_fitting.py (3372 lines) and convolve_fitting.py to a separate
task — too much surface area for one pass.

Picked Option 1 from the architecture menu: functional core + dataclass shells.
Pure-function PSF builders (ssb_psf, adf_psf, icom_psf, epie_psf) over a
Probe + Aberrations + Grid data triple. Drops ProbeParameters / kwargs
factories / ABC + 4 leaky subclasses / dual sign-convention path. PyTorch
backend so PSFs compose with the existing torch fit loops without
host↔device round-trips.

Spec written to docs/superpowers/specs/2026-05-03-instruments-optics-refactor-design.md.
Awaiting user review before kicking off the implementation plan.

### EARS — Progress (2026-05-03 23:34)
<!-- concepts: optics-refactor, aberrations, dataclass-design -->
Implementing the optics refactor approved at 23:24. Bottom-up: constants.py
done, aberrations.py done. Aberrations is a frozen dataclass with named
polar-coefficient fields (C10, C12, phi12, ...) plus an aliasing __init__
that accepts friendlier names (defocus → -C10, Cs → C30, etc.). One sign
convention: defocus = -C10, abtem-aligned. from_legacy_list() handles the
back-compat shim path. Stripped per-instance string metadata (Krivanek,
Haider, Description) — that was class info masquerading as instance state.

Next: chi.py and grad_chi.py PyTorch primitives.

### EARS — Progress (2026-05-03 23:45)
<!-- concepts: optics-refactor, psf-builders, abtem-equivalence -->
optics/ package fully built: constants, aberrations, chi, envelopes,
aperture, grid, probe, psf, __init__. Smoke-tested end-to-end — all four
PSFs (ssb, adf, icom, epie) produce centered, normalized output with
sensible values for both unaberrated and severely-aberrated probes.

Test suite tests/test_optics_vs_abtem.py written: parameterized chi,
temporal envelope, spatial envelope tests against abtem reference (closed
forms inlined since abtem needs Python 3.11+). Plus finite-difference
gradient check, sign-convention sanity, and probe wave normalisation/
centering.

Next: run tests, build _legacy shim, migrate qem/fit/, then delete old
probe.py + ctf.py.

### EARS — Progress (2026-05-04 00:04)
<!-- concepts: optics-refactor, legacy-shim, fft-conventions -->
Optics package + legacy shim wired into qem/instruments/__init__.py.
All 35 legacy CTF tests + 20 new abtem-equivalence tests pass; full
test suite at 155 passed / 1 skipped.

PSF helpers (calculate_psf_width, extract_psf_from_atom_image) moved to
qem/processing/psf.py — they're image-analysis utilities, not optics.

Next: migrate qem/fit/ consumers off the deprecated CTF classes and the
ProbeParameters/create_probe_parameters/calculate_psf_width imports from
qem.instruments. Then delete old probe.py + ctf.py.
