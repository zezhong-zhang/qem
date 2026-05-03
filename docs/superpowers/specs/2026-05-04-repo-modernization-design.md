# QEM Repo Modernization — Phased Refactor Design (2026-05-04)

> Modernize the entire `qem/` package for Python ≥3.11, drop the
> Keras-on-PyTorch compatibility shim, decompose the
> 3372-line `ImageFitting` god-object, and clean up the supporting
> utilities.  Breaking changes allowed; backward-compat shims (e.g.
> `qem/instruments/_legacy.py` from the prior optics refactor) are
> dropped at the end.

## Why phased

The repo is **22 445 LoC across 11 subpackages**.  A single mega-PR is
unreviewable, can't keep tests green throughout, and any rollback would
be all-or-nothing.  Each phase below is its own commit/PR, ends with the
test suite green, and is independently reviewable.

## Repo snapshot (today)

| Subsystem | LoC | Notes |
|---|---:|---|
| `qem/fit/image_fitting.py` | **3 372** | Single god-class with 89 methods.  Heart of the fitting workflow.  Uses keras-shim. |
| `qem/io/dm.py` | **1 127** | Gatan DM3/DM4 parser.  Old-style `class read_DM(object)`. |
| `qem/analysis/gaussian_mixture_model.py` | **1 334** | Multiple classes; mostly numpy/sklearn. |
| `qem/utils/memory_optimization.py` | **761** | 4 over-engineered classes — likely premature optimization. |
| `qem/fit/ptychography_optimization.py` | **715** | Uses keras-shim. |
| `qem/fit/linear_solver.py` | **713** | Uses keras-shim. |
| `qem/fit/point_potential.py` | **591** | Uses keras-shim. |
| `qem/instruments/_legacy.py` | **564** | Back-compat shim for the previous optics refactor. |
| `qem/fit/convolve_fitting.py` | **954** | Uses keras-shim. |
| `qem/utils/torch_compat.py` | **254** | The keras shim itself.  Target for deletion. |
| `qem/utils/logging_config.py` | **341** | Custom formatters + PerformanceTracker. |
| `qem/instruments/optics/` | **1 180** | Already modern (just refactored). |
| Other | ~10 000 | Detector, tilt, schema, processing, IO, visualization, analysis, optimizers, archive, benchmarks, scalebar, app. |

**Keras-shim consumers:** 11 files in `qem/`, 5 in `tests/`.  Dropping
the shim is a coordinated edit across the entire `qem/fit/` subsystem.

## Goals

1. **Modern Python** (≥ 3.11): type hints everywhere, `dataclass(slots=True,
   frozen=True)` where possible, `pathlib.Path` over `os.path`, f-strings
   over `%`/`.format`, structural pattern matching where it clarifies,
   `match`/`case` for enums, `tomllib` for config.
2. **Clear architecture, flat layout**: every module is at most two
   directory levels deep (``qem/<package>/<file>.py``).  No
   sub-subpackages.  ``ImageFitting`` decomposed into focused sibling
   modules under ``qem/fit/``, not into a new ``qem/fit/image_fitting/``
   subdirectory.  Short, role-named files (``fitter.py``, ``solver.py``,
   ``loop.py``); class and method names follow the same rule.
3. **Performance**: PyTorch-only (no Keras shim, no numpy↔torch
   round-trips inside fit loops), GPU-ready throughout, lazy/cached PSF
   construction, memory-efficient sparse design matrices.

## Non-goals

- Rewriting `qem/io/dm.py` from scratch (DM file parsing is too touchy
  to redo without a battery of regression tests against real files;
  modernize style only).
- Touching `qem/app.py` (Streamlit GUI lives outside the library
  surface).
- Adding new features.  This is a refactor, not a redesign.

## Phased plan

Each phase ends with **all tests green**, a **single commit**, and
**leaves the repo in a working state**.

### Phase 0 — Cleanup, flatten, ground rules (low risk, ~1 day)

- Delete `qem/archive/` (dead code).
- Delete `qem/instruments/_legacy.py` and update tests/examples to use
  the new optics API directly.  Now that we're committing to "no
  back-compat", the shim earns its keep no longer.
- **Flatten `qem/instruments/optics/*.py` up into `qem/instruments/`**
  so every module is at most two levels deep.  After:
  ```
  qem/instruments/
    __init__.py
    aberrations.py     (was optics/aberrations.py)
    aperture.py        (was optics/aperture.py)
    chi.py             (was optics/chi.py)
    constants.py       (was optics/constants.py)
    detector.py        (unchanged)
    envelopes.py       (was optics/envelopes.py)
    grid.py            (was optics/grid.py)
    probe.py           (was optics/probe.py — name reused)
    psf.py             (was optics/psf.py)
    tilt.py            (unchanged)
    wave.py            (unchanged)
  ```
  All ~10 internal consumers (`qem/fit/*`, tests, examples) updated.
- Bump the project Python floor to 3.11 in `pyproject.toml`.
- Add a pre-commit config: `ruff check`, `ruff format`, `mypy --strict`
  on the modules we'll touch in later phases (gradual rollout).
- Replace `os.path.*` with `pathlib.Path` in the obvious places (only
  where it doesn't cascade into other changes).
- Replace `%`-formatting and `.format()` with f-strings.

**Test impact:** 155 tests today must stay 155 passing.  Public API
changes: `_legacy.py` gone, optics imports change from
`qem.instruments.optics.foo` → `qem.instruments.foo`.

**Affected files:** ~20.  ~−700 / +50 LoC.

### Phase 1 — Drop the keras shim, switch to pure PyTorch (~2 days)

- Delete `qem/utils/torch_compat.py`.
- Delete `qem/utils/backend.py::safe_convert_to_tensor` (only existed to
  hide the shim import).
- For every consumer (`qem/fit/{linear_solver,point_potential,
  ptychography_optimization,image_fitting,model,convolve_fitting}.py`,
  `qem/utils/{params,memory_optimization}.py`,
  `qem/schema/validation.py`, benchmarks, tests):
  - `keras.Model` → `torch.nn.Module`.
  - `keras.layers.X` → `torch.nn.X`.
  - `keras.ops.Y` → `torch.Y` / `torch.fft.Y`.
  - `keras.optimizers.X` → `torch.optim.X`.
  - `keras.callbacks` (mostly EarlyStopping / progress) → a small,
    explicit set in a new `qem/fit/_loop.py`.
  - `model.fit(...)` → an explicit training loop in `qem/fit/_loop.py`
    (~150 lines): for-epoch / for-batch, autograd, early-stopping
    callback, optional tqdm.  No more hidden keras semantics.
- A new `qem/utils/torch_io.py` (~50 lines) for the small surface that
  every module needs: `as_tensor`, `to_numpy`, `device_of`,
  `default_dtype()`, `set_default_device()`.

**Test impact:** every keras-shim test in `tests/test_model_api.py`,
`tests/test_model.py`, `tests/test_basic.py`, `tests/test_utils.py`,
`tests/test_linear_solver.py`, `tests/test_image_fitting.py` is
rewritten / verified.  All pass.

**Affected files:** ~20.  ~−500 / +400 LoC net.

### Phase 2 — Decompose `qem/fit/image_fitting.py` (~3 days)

3372 lines, 89 methods, one class.  Decomposed into **flat sibling
modules under `qem/fit/`**, no new subpackage.  Target layout:

```
qem/fit/
  fitter.py        # the orchestrating Fitter class (was ImageFitting) — ≤ 700 lines
  fit_params.py    # FitParameters dataclass (replaces dict-of-tensors)
  fit_loop.py      # standard + edge-correction training loops
  refine.py        # refine_center_of_mass, refine_local_max  (extends existing)
  peaks.py         # find_peaks, dedup, edge-removal
  loss.py          # loss(), boundary penalty, adaptive edge loss, visibility
  bg2d.py          # 2D background enable/disable/optimize
  ...              # + existing siblings: convolve.py, ptycho.py, solver.py,
                   # potential.py, model.py, voronoi.py, background.py
```

- Every file ≤ 700 lines.
- Class rename: `ImageFitting` → `Fitter`.  File rename:
  `image_fitting.py` → `fitter.py`.  `qem.fit.ImageFitting` re-exported
  for one-line back-compat (the design doc says "drop backward
  support", but `Fitter` is short enough that not having an alias
  isn't painful — re-export removed at end of phase 5).
- `dict`-of-tensor parameter bags become a `FitParameters` dataclass
  with named fields: `pos_x`, `pos_y`, `height`, `width`, `background`,
  optional `bg_2d_*`.
- `predict()` and `loss()` get clean type signatures.
- All numerical behavior preserved; tests stay green.

**Test impact:** `tests/test_image_fitting.py` keeps passing through
the new `qem.fit.fitter.Fitter` (with a transitional
`ImageFitting = Fitter` re-export in `qem/fit/__init__.py`).

**Affected files:** ~12.  Net LoC roughly the same; the win is
decomposition + clearer names.

### Renames carried out across the phases

To keep names short and roles clear:

| Old | New | Phase |
|---|---|---|
| `qem/instruments/optics/*.py` | `qem/instruments/*.py` | 0 |
| `qem/instruments/_legacy.py` | (deleted) | 0 |
| `qem/utils/torch_compat.py` | (deleted) | 1 |
| `qem/utils/backend.py` | `qem/utils/tensors.py` | 1 |
| `qem/fit/image_fitting.py` | `qem/fit/fitter.py` | 2 |
| `qem/fit/convolve_fitting.py` | `qem/fit/convolve.py` | 2 |
| `qem/fit/ptychography_optimization.py` | `qem/fit/ptycho.py` | 2 |
| `qem/fit/linear_solver.py` | `qem/fit/solver.py` | 2 |
| `qem/fit/point_potential.py` | `qem/fit/potential.py` | 2 |
| `qem/utils/memory_optimization.py` | `qem/utils/memory.py` | 3 |
| `qem/utils/logging_config.py` | `qem/utils/log.py` | 3 |
| `qem/utils/periodic_table.py` | `qem/utils/elements.py` | 3 |
| `qem/visualization/` | `qem/viz/` | 3 |
| `qem/visualization/add_coordinate.py` | `qem/viz/coords.py` | 3 |

Class renames:

| Old | New |
|---|---|
| `ImageFitting` | `Fitter` |
| `ConvolutionFitting` | `ConvFit` |
| `PtychographyFitting` | `PtychoFit` |
| `ADFConvolutionFitting` | `AdfConvFit` |
| `PtychographyOptimizer` | `PtychoOptimizer` |
| `LinearSystemSolver` | `LinSolver` |
| `BackgroundEstimator` | `Background` |
| `MemoryOptimizer*` family | (mostly deleted, see Phase 3) |

### Phase 3 — Trim utils (~1 day)

- `memory_optimization.py` (761 lines, 4 classes): keep only what's
  used.  My grep last hour found `BatchMemoryOptimizer`,
  `ChunkedProcessor`, `SparseMatrixOptimizer`, `MemoryMonitor` are
  imported via `qem/processing/__init__.py`'s wildcard but mostly never
  called — verify and prune.  Target: ≤ 200 lines.
- `logging_config.py` (341 lines): replace with thin wrapper around
  stdlib `logging` + `tqdm.contrib.logging`.  Drop the custom
  `PerformanceTracker` (5-line context manager around `time.perf_counter()`
  is enough).  Target: ≤ 80 lines.
- `qem/optimizers/lbfgs.py` (95 lines): inline if it's a thin wrapper
  around `torch.optim.LBFGS`; otherwise modernize in place.

**Affected files:** ~5.  ~−800 / +200 LoC.

### Phase 4 — Modernize `qem/io/dm.py` style (~1 day)

- Replace `class read_DM(object)` and `class write_DM(object)` with
  modern classes.
- Type hints, `pathlib.Path`, dataclasses for header structures.
- **Do not change parsing logic** — gate every change with a regression
  test against a known DM3 + DM4 file.  If we don't have one, the
  phase reduces to style-only changes (no logic touched).

**Affected files:** 1-2.  Net LoC similar.

### Phase 5 — Polish (~½ day)

- `qem/analysis/gaussian_mixture_model.py`: type hints + tighter API
  (the 1334-line file is multiple classes — leave their internals,
  modernize the surface).
- `qem/io/dm.py`: only if Phase 4 didn't already get to it.
- `__all__` discipline across every public package.
- README + CLAUDE.md update if they reference the deleted shim.

## Test strategy

- Every phase ends with `pytest tests/` green (155 today + any new
  tests added during phases).
- Phase 1 (keras→torch) gets a temporary parity check: numerical
  equality between the old keras-shim output and the new pure-torch
  output for one full `fit_positions` end-to-end run, asserted to ≤
  1e-5 relative.  Removed before final commit.
- Phase 2 (image_fitting decomposition) is purely mechanical; if
  numerical behavior changes, it's a bug.

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 1 silently changes optimization semantics | Parity test + keep loss/optimizer choices identical bit-for-bit. |
| Phase 2 breaks the public `ImageFitting` API | Keep `from qem.fit.image_fitting import ImageFitting` working; only the internals move. |
| Phase 4 corrupts DM file parsing | Only style changes unless we have a regression test. |
| One phase introduces a typing bug that mypy doesn't catch | Each phase ends with a manual smoke run of one representative example script + the test suite. |
| Auto-mode runs ahead | Each phase commits separately with a clear commit message; user can stop, review, redirect at any phase boundary. |

## Estimates

| Phase | Effort | Net LoC |
|---|---:|---:|
| 0 — cleanup & ground rules | ½ day | −200 / +50 |
| 1 — drop keras shim | 2 days | −500 / +400 |
| 2 — decompose image_fitting | 3 days | ±0 (decomposition) |
| 3 — trim utils | 1 day | −800 / +200 |
| 4 — modernize io/dm.py style | 1 day | ±0 |
| 5 — polish | ½ day | −100 / +50 |
| **Total** | **~8 days** | **~−1 600 net** |

## Acceptance

1. `pytest tests/` green at every phase boundary.
2. No remaining imports of `qem.utils.torch_compat` after Phase 1.
3. No file in `qem/` exceeds 800 lines after Phase 2.
4. `mypy --strict qem/` clean on all modules touched in any phase.
5. `ruff check qem/` clean.
6. Repo total LoC drops by ≥ 1500.

## Out of scope (future)

- Migrating `qem/app.py` (Streamlit GUI) to a modern framework.
- Replacing `qem/io/dm.py` with `hyperspy` or `gatan-fileformats`
  (would be a big external dependency choice).
- CUDA-specific kernel optimizations beyond using torch ops on the
  default device.
