# Linus-style review of `qem/`

> "I'm not going to mince words. Some of this is good. Some of it would
> make me reach for the [DELETE] key."

Written 2026-05-04, against tip-of-master after the 7-phase
modernisation refactor (commits `786ea4c..088d586`).

## Top-3 problems, in order of severity

### 1. `Fitter` is a god class. Period.

[`qem/fit/fitter.py`](../../../qem/fit/fitter.py) — **3354 lines, 87
def/class statements, 1 class.** Method inventory by domain:

| Concern | # methods | Should live in… |
|---|---:|---|
| Init / save / load / memory-monitor | ~10 | `fitter.py` core — keep |
| Model selection / build | ~3 | `fitter.py` — keep |
| Loss / boundary penalty / edge loss / window | ~7 | `loss.py` |
| Peak finding / dedup / edge removal / refine_com / refine_local_max | ~9 | `peaks.py` |
| Background (1D + 2D + scale + estimator) | ~6 | already have `background.py` — move them |
| **Plotting** | **11** | `plot.py` |
| Optimisation (`fit_global`, `fit_stochastic`, `fit_with_edge_correction`, `optimize`, `_plot_progress`) | ~5 | `loop.py` (already have `_loop.py`) |
| Linear estimator (`linear_estimator`, `_prepare_target_vector`, `_process_solution`) | ~3 | move into `solver.py` |
| GMM + atom counting (`estimate_atom_counts_with_gmm`, `_plot_gmm_results`, `plot_atom_count_map`, `integrate_gmm_with_crystal_analyzer`, `update_all_regions_with_gmm`, `export_gmm_updated_structure`) | **6** | `gmm.py` (or call out to `qem.analysis.gmm`) |
| Crystal/lattice/region (`map_lattice`, `assign_region_label`, `update_region_analyzers`, `select_atoms`, `view_3d`) | 5 | `qem.analysis.crystal_analyzer` already exists — call into it, don't duplicate |
| Domain analysis (`estimate_complex_domains`, `_separate_vacuum_and_sample`, `_identify_domain_boundaries`, `_create_polygon_enclosures`, `_plot_domain_analysis`) | **5** | `domains.py` — separate concern entirely |
| Voronoi (`fit_voronoi`, `voronoi_integration`, `plot_voronoi_…`) | 4 | already have `voronoi.py` — move them |
| Misc properties / calibrate / convert | ~10 | keep |

**This isn't an aesthetic gripe.** A god class means: every change
ripples; tests are coupled; you can't swap an optimiser without
dragging in `matplotlib`; `from qem.fit.fitter import Fitter` pulls in
`h5py`, `matscipy`, the entire viz stack, the GMM module, ASE, and
`crystal_analyzer`. **Import time alone matters** for a CLI /
Jupyter-startup case.

The fix isn't another big refactor — it's: extract one concern at a
time as a free function or a small data-class, reduce `Fitter` to
**orchestration only** (state + delegation). Target: `fitter.py` ≤ 800
lines, no `plot_*` / no `_plot_*` / no `estimate_complex_domains` on
the class.

### 2. Defensive coding has metastasised

```
fitter.py: 49 try/except/raise statements, 47 None-checks
```

[`qem/fit/fitter.py:106-129`](../../../qem/fit/fitter.py#L106-L129) —
`__init__` wraps a literal `try: self.image = image; self.dx = dx; …`
in `try / except Exception as e: logging.error; raise`. **You're
catching `Exception`, logging it, and re-raising.** That gives you
nothing the traceback wouldn't have given you. Delete the try/except,
delete the per-bool `isinstance` checks.

```python
# What's there:
try:
    self.image = image
    self.dx = dx
    ...
    if not isinstance(units, str):
        raise ValueError("Units must be a string")
    if len(units) == 0:
        raise ValueError("Units cannot be empty")
    for param_name, param_value in [("same_width", same_width), ...]:
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean, got {type(param_value)}")
except Exception as e:
    logging.error(f"Fitter initialization failed: {str(e)}")
    raise
```

```python
# What it should be:
self.image = image
self.dx = dx
self.units = units            # type-annotated; if you want strictness, mypy will tell you
self.same_width = bool(same_width)
self.pbc = bool(pbc)
self.fit_background = bool(fit_background)
```

If a caller passes a dict for `units`, the *first time it's used* will
raise a clear `AttributeError` at the call site. That's the right
place — not a re-raised wrapper at construction time.

The rest of [`qem/fit/validation.py`](../../../qem/fit/validation.py)
(462 lines, mostly *more* of these checks: `max_size: int = 5000,
max_memory_mb: int = 1000`) is the same anti-pattern at scale. **Real
validation belongs at API boundaries, not for every Python `bool`
argument.** This file's `ValidationError` even bypasses the
`qem.utils.exceptions` hierarchy you just unified.

### 3. The "memory-optimisation" subsystem is mostly dead code

[`qem/utils/memory.py`](../../../qem/utils/memory.py) — **757 lines, 4
classes**. Real usage:

| Class | Imported by | Actually called by |
|---|---|---|
| `MemoryMonitor` | `fitter.py` | yes (2 sites: `__init__` + `enable_memory_monitoring`) |
| `BatchMemoryOptimizer` | `fitter.py` | **0 sites** (just `from … import …`) |
| `ChunkedProcessor` | `fitter.py` | **0 sites** |
| `SparseMatrixOptimizer` | `fitter.py` | **0 sites** |

That's **3 classes of premature optimisation, ~600 lines, zero
callers.** Delete them. Keep `MemoryMonitor` if you actually use it,
but `psutil`-based RSS polling around an Adam loop is theatre — the
real memory hog is your tensor allocations, and
`torch.cuda.memory_allocated()` is one line.

---

## Top-3 performance opportunities

### A. Stop bouncing between numpy and torch

[`qem/fit/fitter.py`](../../../qem/fit/fitter.py) has **42**
`safe_convert_to_numpy(...)` / `safe_convert_to_tensor(...)` calls.
Every one of these is at minimum a `.detach().cpu().numpy()`
(allocation, host transfer, dtype convert). In a tight loop they're a
non-trivial cost.

Concrete examples:

- `predict()` returns torch → `safe_convert_to_numpy(self.predict(...))`
  → store as `self.prediction = ndarray` → next call to `predict`
  rebuilds tensors → loop.
- `optimize()` returns `dict[str, np.ndarray]` (per its annotation)
  but `model.get_params()` actually returns torch tensors. The
  annotation lies; downstream code re-`safe_convert_to_tensor`s them.

**Fix:** keep state on the *device* the model parameters live on.
Convert to numpy *only* at IO boundaries (`save`, `plot`,
`return-to-user`). Add ONE `Fitter.detach()` method that snapshots
everything to numpy on demand.

This is the single biggest perf win available without algorithmic
changes. Easily 2× on a Mac laptop (where MPS↔CPU is expensive); much
more on CUDA.

### B. `_sum_local` allocates the world every call

[`qem/fit/model.py:161-199`](../../../qem/fit/model.py#L161-L199):

```python
max_width = float(np.max(to_numpy(self.input_params["width"])))
window_size = int(max_width * 4)
coords = torch.arange(-window_size, window_size + 1, dtype=x_grid.dtype, device=x_grid.device)
local_x, local_y = torch.meshgrid(coords, coords, indexing="xy")
...
canvas = torch.zeros_like(x_grid).reshape(-1)
canvas = canvas.scatter_add(0, flat_indices, masked_peaks.reshape(-1))
```

Every fit step:

1. `to_numpy()` round-trip just to compute `max(width)` — use
   `self.width.detach().max().item()`.
2. `torch.meshgrid(coords, coords)` re-allocated every call — cache it
   on the model (it only depends on `window_size`).
3. `torch.where(in_bounds, local_peaks, torch.zeros_like(local_peaks))`
   allocates a zero tensor of `(2W+1)²·N` shape just to multiply by
   0/1 — a `local_peaks * in_bounds` does the same, no extra
   allocation.
4. `flat_indices = global_y_safe.reshape(-1) * w + global_x_safe.reshape(-1)`
   — for `N` atoms, `(2W+1)² · N` elements. For 5000 atoms × 8² window
   = 320 000 indices per step. Cache the per-atom local windows; only
   positions vary.

A modest cleanup here is **2–5× faster `fit_global` on the typical
case**.

### C. `fit_loop` rebuilds `image_tensor` and the LR scheduler from scratch every call

[`qem/fit/fitter.py:1675-1685`](../../../qem/fit/fitter.py#L1675-L1685):

```python
image_tensor = torch.unsqueeze(image_tensor, 0)
x_grid = torch.unsqueeze(self.x_grid, 0)
y_grid = torch.unsqueeze(self.y_grid, 0)
model_inputs = [x_grid, y_grid]
```

Three new tensors per `optimize()` call — fine for one call, wasteful
when `fit_global` calls into stochastic batches. `self.x_grid` should
already be precomputed at `Fitter.__init__` with the correct dims
(currently 2-D; just store the unsqueezed 3-D version too, or expand
on-the-fly with `.unsqueeze(0)` and trust torch's view semantics — it's
free).

The fit loop in [`qem/fit/_loop.py:67`](../../../qem/fit/_loop.py#L67)
**snapshots the whole model state_dict every epoch where loss
improves**. For a 5000-atom fit, that's `5 × 5000 × 4 bytes = 100 KB
per snapshot × maybe 50 improvements = 5 MB of churn**, probably
triggering allocator fragmentation. Snapshot only every N epochs, or
only at the end (Adam's last-step output is usually fine).

---

## Smaller things worth fixing in one sitting

1. **`qem/fit/solver.py` (713 lines) vs `qem/fit/sparse.py` (131
   lines)** still overlap. `solver.py` has *its own*
   `LinearSystemSolver` with a `try/except → fall back to a
   try/except` pattern (lines 374–440). The 60-line "memory-aware
   strategy selection" with `psutil` polling at every solve call is
   overhead, not optimisation. Replace with: try direct, on
   `MatrixRankWarning`/MemoryError fall back to iterative. Two lines,
   no `psutil`. The `sparse.py` already did it right.

2. **`qem/utils/log.py` (341 lines)** still has `StructuredFormatter`,
   `UserFriendlyFormatter` (with emojis 🐛ℹ️⚠️❌🚨 in the actual log
   output), `QEMLogger`, `PerformanceTracker`, `TqdmLoggingHandler`.
   Nobody is sending QEM logs to ELK; you're a numerical library.
   **Replace with 30 lines of stdlib `logging.basicConfig` + a
   `tqdm.contrib.logging` import**. Emojis in logs are friendly until
   your CI parses them and chokes.

3. **`qem/fit/lbfgs.py`** wraps `torch.optim.LBFGS` with a custom
   95-line API. The `make_optimizer("lbfgs", …)` path in `_loop.py`
   doesn't even cover it — there's a separate code path in
   `Fitter.optimize` (lines 1687-1710). Either fold L-BFGS into the
   standard `_loop.py` path with `closure=` (PyTorch's idiom), or
   admit you don't use L-BFGS and delete it.

4. **Validation hierarchy is muddled.**
   [`qem/fit/validation.py:12`](../../../qem/fit/validation.py#L12)
   defines `ValidationError(ValueError)` *separately* from
   [`qem/utils/exceptions.py`](../../../qem/utils/exceptions.py)
   (which already has a `ValidationError`). Two `ValidationError`s.
   Pick one.

5. **`Fitter.optimize` is a `dict` factory.** Annotation says `->
   dict[str, NDArray[Any]]`. Reality returns torch tensors via
   `model.get_params()`. Type the return correctly or convert; don't
   lie.

6. **`route_b_joint_ls.py` does explicit projected-gradient descent**
   ([line 158](../../../qem/fusion/route_b_joint_ls.py#L158): `for
   iteration in range(self.max_iter)`). It's 200+ lines of hand-rolled
   SGD with TV regularisation, all numpy, no autograd. **This belongs
   on torch with `torch.autograd.functional.grad` or just an Adam
   loop** — exactly the same primitive you built for `fit_global`.
   Reuse `qem.fit._loop.fit_loop`.

7. **3 notebooks (`introduction.ipynb`, `STO.ipynb`,
   `benchmark.ipynb`) and 2 example .py files** have **zero CI
   coverage**. They were broken twice during this refactor and you
   didn't notice until I went looking. Add `pytest --nbmake
   examples/*.ipynb` (gated on test data presence) or smoke-test the
   .py examples in CI.

8. **`qem/__init__.py`** still does `from .fit.fitter import Fitter`.
   That triggers the entire `fitter.py` import-time dependency cascade
   for `import qem`. Move `Fitter` into a lazy attribute (`__getattr__`
   at module level).

---

## What's actually good

- The **`qem/optics/`** package is genuinely clean. `chi.py` +
  `envelopes.py` + `psf.py` + tests against a closed-form abtem
  reference is exactly the right pattern. Don't touch it.
- **`qem/fit/_loop.py`** is the right size (130 lines) and the right
  shape — explicit loop, named knobs, no callbacks framework.
- **`qem/utils/tensors.py`** is 88 lines. After replacing
  `qem/utils/{params,backend}.py` re-exports with direct calls to it,
  you can delete those wrapper files entirely.
- The **flat layout** is doing its job — every module is one `from
  qem.<pkg>.<file> import …` away.

---

## Concrete punch list (smallest unit of useful work first)

| # | Action | Files | LoC delta |
|---|---|---|---:|
| 1 | Delete `BatchMemoryOptimizer`, `ChunkedProcessor`, `SparseMatrixOptimizer` from `memory.py`, drop their imports in `fitter.py` | 2 | **−600** |
| 2 | Delete the `try/except → log → raise` and per-bool `isinstance` checks in `Fitter.__init__` | 1 | −20 |
| 3 | Cache `meshgrid` + `coords` in `ImageModel._sum_local`; replace `to_numpy(width).max()` with `self.width.detach().max().item()` | 1 | −5 |
| 4 | Replace `qem/utils/log.py` with ≤ 50-line stdlib `logging.basicConfig` wrapper | 2 | **−290** |
| 5 | Reconcile `qem/fit/validation.py::ValidationError` with `qem/utils/exceptions.py::ValidationError` (delete the duplicate, keep the hierarchy one) | 2 | −20 |
| 6 | Fold L-BFGS into `_loop.fit_loop` via `closure=`; delete `qem/fit/lbfgs.py` | 2 | −90 |
| 7 | Make `qem.Fitter` lazy via `__getattr__` so `import qem` doesn't pull in `fitter.py`'s ~40 dependencies | 1 | +5 |
| 8 | Add `pytest --nbmake examples/*.ipynb` to CI (or a smoke-test that imports + runs the first cell) | 1 | +20 |
| 9 | **Decompose `Fitter`**: extract `plot_*` to `qem/fit/plot.py`, `estimate_complex_domains` family to `qem/analysis/domains.py`, `*_gmm_*` family to delegation calls into `qem/analysis/gaussian_mixture_model.py` | many | ~0 (just moves) |
| 10 | Move the projected-gradient loop in `route_b_joint_ls.py` onto torch + `_loop.fit_loop` | 1 | −80 |

Items 1–8 are **one-line-each-style** changes; you could close them in
an afternoon and shed roughly **−1000 LoC + measurable speedup**.
Item 9 is the multi-day project. Item 10 is a quality win for the
fusion code.

---

## What I'd refuse to merge as-is

- The `try: self.image = image; ... except Exception: logging.error;
  raise` pattern. **It's worse than no try/except.**
- Keeping all 3 dead memory-optimiser classes when the import of them
  is the only thing keeping them alive.
- Keeping two `ValidationError` classes in two different modules.
- `Fitter` continuing to grow another method in 6 months because
  nobody pushed back.

If you want to act on this, the highest **leverage:cost** items are #1
(delete dead memory code), #4 (trim log.py), and #2 (gut the defensive
__init__). All three together: ~30 min, **~−900 LoC**, no behaviour
change.
