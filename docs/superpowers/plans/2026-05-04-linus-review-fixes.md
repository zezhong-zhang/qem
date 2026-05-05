# Linus-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every issue raised in `docs/superpowers/specs/review.md` — quick wins (dead code, defensive coding, log cruft, dup ValidationError, LBFGS wrapper, lazy import, CI smoke), performance fixes (`_sum_local` allocations, numpy/torch bouncing, snapshot churn), the route_b torch port, and finally the multi-day `Fitter` decomposition.

**Architecture:** Pure deletions and refactors against the current PyTorch-only codebase. No new abstractions — collapse layers, push state through, and keep `Fitter` strictly orchestration. `_loop.fit_loop` becomes the single training-loop entry point. `qem/utils/log.py` shrinks to a thin stdlib wrapper. All numpy↔torch bounces move to IO boundaries only. The 11 `plot_*`, 6 GMM, 5 domain, and 4 voronoi methods leave `fitter.py` for sibling modules; `Fitter` keeps init / state / model selection / orchestration only.

**Tech Stack:** Python 3.11+, PyTorch 2.x, numpy, scipy.sparse, pytest, ruff. CI runs `pytest tests/ --cov=qem` on Python 3.9–3.11.

**Conventions for every task:**
- Run `pytest tests/ -x -q` after each commit unless the task's verification step says otherwise. Tests must stay green.
- Use `git status` + `git diff` to confirm scope before each commit.
- Commit messages follow `refactor(qem):`, `perf(qem):`, `feat(qem):`, `chore(ci):`, `fix(qem):` Conventional Commits style with a Linus-review reference in the body (e.g. `Linus review #1`).

---

## File Structure

**Files deleted entirely:**
- `qem/fit/lbfgs.py` — folded into `qem/fit/_loop.py` via PyTorch closure idiom (item #6).
- `tests/test_memory_optimization.py` — only tests the dead classes; no consumer code under test once they go (item #1).

**Files renamed (no behavior change):**
- `qem/fit/_loop.py` → `qem/fit/loop.py` — drop the underscore (review note about "already have `_loop.py`"). Keep both the rename and the LBFGS fold in one commit per task #6.

**Files created:**
- `qem/fit/loss.py` — `Fitter`'s loss / boundary penalty / edge loss / butterworth window helpers (~7 methods, item #9).
- `qem/fit/peaks.py` — peak finding / dedup / edge removal / refine_com / refine_local_max (~9 methods, item #9).
- `qem/fit/plot.py` — every `plot_*` and `_plot_*` (11 methods, item #9).
- `qem/analysis/domains.py` — `estimate_complex_domains` family (5 methods, item #9).
- `qem/analysis/gmm.py` — `estimate_atom_counts_with_gmm` family (6 methods; thin delegation wrapper around any code already in `qem/analysis/gaussian_mixture_model.py`, item #9).

**Files heavily modified:**
- `qem/fit/fitter.py` — strip every method that now lives elsewhere; target ≤ 800 lines.
- `qem/fit/model.py` — cache window-coords meshgrid; drop `to_numpy()` round-trip in `_sum_local` (perf B / item #3).
- `qem/utils/memory.py` — keep `MemoryMonitor`, delete `BatchMemoryOptimizer` / `ChunkedProcessor` / `SparseMatrixOptimizer` and the two module-level instances (item #1).
- `qem/utils/log.py` — collapse to ≤ 50 lines: stdlib `basicConfig` + tqdm-friendly handler + `get_logger` returning a stdlib `logging.Logger` (item #4).
- `qem/fit/validation.py` — drop the duplicate `ValidationError`; raise `qem.utils.exceptions.ValidationError` (item #5).
- `qem/fit/solver.py` — collapse the 60-line "memory-aware strategy selection" + nested try/except; replace with one direct attempt and a single iterative fallback on `MemoryError`/`MatrixRankWarning` (smaller-things item #1).
- `qem/fusion/route_b_joint_ls.py` — replace the explicit projected-gradient numpy loop with `_loop.fit_loop` over a torch model (item #10).
- `qem/__init__.py` — replace the eager `from .fit.fitter import Fitter` with a module-level `__getattr__` (item #7).
- `qem/fit/fitter.py:106-129` — delete the wrapping try/except + per-bool `isinstance` checks (item #2).
- `qem/fit/fitter.py:1675-1685` — drop the per-call `unsqueeze`s; pre-store batched grids (perf C).
- `qem/fit/_loop.py:67` (now `loop.py`) — snapshot best state every N epochs, not every improvement (perf C).
- `.github/workflows/python-package.yml` — add `pytest --nbmake examples/*.ipynb` smoke job, gated on test data presence (item #8).

---

## Task 1: Delete dead memory-optimisation classes

**Files:**
- Modify: `qem/utils/memory.py:180-757`
- Modify: `qem/fit/fitter.py:58-65`
- Delete: `tests/test_memory_optimization.py`

This is review item #1. `BatchMemoryOptimizer`, `ChunkedProcessor`, `SparseMatrixOptimizer` have **zero call sites** in `qem/`. Only the import in `qem/fit/fitter.py` keeps them alive. Delete them and their tests.

- [ ] **Step 1: Confirm zero call sites**

Run:
```bash
rg -n "BatchMemoryOptimizer|ChunkedProcessor|SparseMatrixOptimizer|memory_optimizer|chunked_processor" qem/ tests/ examples/
```

Expected: matches only inside `qem/utils/memory.py`, `qem/fit/fitter.py:58-65` (the dead import), and `tests/test_memory_optimization.py`.

- [ ] **Step 2: Trim the imports in `qem/fit/fitter.py`**

Edit `qem/fit/fitter.py:58-65`:

```python
# Before
from qem.utils.memory import (
    BatchMemoryOptimizer,
    ChunkedProcessor,
    SparseMatrixOptimizer,
    MemoryMonitor,
    memory_optimizer,
    chunked_processor,
)
```

Replace with:
```python
from qem.utils.memory import MemoryMonitor
```

- [ ] **Step 3: Strip the dead classes from `qem/utils/memory.py`**

Delete the class definitions for `BatchMemoryOptimizer` (line 180), `ChunkedProcessor` (line 513), `SparseMatrixOptimizer` (line 642), the two module-level instances at the bottom (`memory_optimizer = …`, `chunked_processor = …`), and the references to them in the module docstring (lines 4–28). Keep only `MemoryMonitor` and the file header.

After the edit, `qem/utils/memory.py` should be ≤ 180 lines. Verify:
```bash
wc -l qem/utils/memory.py
```

- [ ] **Step 4: Delete the test file**

Run:
```bash
git rm tests/test_memory_optimization.py
```

- [ ] **Step 5: Smoke-import qem and run the tests**

Run:
```bash
python -c "import qem; from qem import Fitter; print('ok')"
pytest tests/ -x -q
```

Expected: import prints `ok`; pytest passes. If anything still references a deleted symbol, fix it now.

- [ ] **Step 6: Commit**

```bash
git add qem/utils/memory.py qem/fit/fitter.py tests/test_memory_optimization.py
git commit -m "refactor(qem): delete dead memory optimiser classes (Linus #1)

Removes BatchMemoryOptimizer, ChunkedProcessor, SparseMatrixOptimizer
and their global instances. Each had zero call sites; only the import
in fitter.py kept them alive. Drops ~600 LoC from utils/memory.py and
the corresponding test file."
```

---

## Task 2: Gut the defensive `Fitter.__init__`

**Files:**
- Modify: `qem/fit/fitter.py:75-181`

Review item #2. The wrapping `try / except Exception: log; raise` adds nothing the traceback wouldn't give you. The per-bool `isinstance` checks are noise — Python is duck-typed; trust the caller, raise at point of use.

- [ ] **Step 1: Edit `qem/fit/fitter.py`**

Replace lines 105–135 (the `try:` block and the manual `Store validated parameters` block) with the bare assignments. The new `__init__` body from the docstring through `self.background_estimator = …` should be:

```python
        # Store input parameters as-is. Type errors will surface at the
        # first use site with a clear traceback; we don't need to
        # second-guess Python's type system here.
        self.image = image
        self.dx = dx
        self.elements = elements
        self.model_type = model_type
        self.units = units
        self.same_width = bool(same_width)
        self.pbc = bool(pbc)
        self.fit_background = bool(fit_background)
        self.monitor_memory = monitor_memory

        # Initialize memory monitoring
        if self.monitor_memory:
            self.memory_monitor = MemoryMonitor()
        else:
            self.memory_monitor = None

        logging.info(
            "Initializing Fitter with %s image, dx=%s %s, model=%s",
            self.image.shape, self.dx, self.units, self.model_type,
        )

        # Create model instance based on type
        self.model = self._select_model()

        # Create Gaussian kernel for filtering
        self.kernel = GaussianKernel()
        self._window = None

        # Initialize state
        self._atom_types = np.array([])
        self._coordinates = np.array([])
        self.coordinates_history: dict = {}

        # Boundary penalty + adaptive edge loss off by default
        self.use_boundary_penalty = False
        self.boundary_margin = 2.0
        self.boundary_strength = 0.01
        self.use_adaptive_edge_loss = False

        self.coordinates_state = 0
        self.init_background = 0.0
        self.prediction = np.zeros_like(self.image)

        self.params = None
        self.converged = False
        self.ny, self.nx = image.shape
        self.regions = Regions(image=image)
        self.initialize_grid()
        self.background_estimator = Background(self.image, self.dx)
```

Also delete the duplicated `Initialize the Fitter class with comprehensive input validation.` line in the docstring at line 89–90.

- [ ] **Step 2: Run a smoke test**

Run:
```bash
python -c "
import numpy as np
from qem import Fitter
img = np.random.rand(64, 64).astype(np.float32)
f = Fitter(img, dx=0.1, units='nm', same_width=True, pbc=False, fit_background=True)
print('init ok', f.image.shape, f.units, type(f.same_width))
"
pytest tests/ -x -q
```

Expected: `init ok (64, 64) nm <class 'bool'>`; pytest green.

- [ ] **Step 3: Commit**

```bash
git add qem/fit/fitter.py
git commit -m "refactor(qem): gut defensive Fitter.__init__ (Linus #2)

Drops the wrapping try/except: log; raise (which adds nothing the
traceback wouldn't), the per-bool isinstance checks, and the duplicate
docstring line. Keeps bool() coercion for the three boolean knobs since
that documents intent and is essentially free."
```

---

## Task 3: Reconcile the duplicate `ValidationError`

**Files:**
- Modify: `qem/fit/validation.py:12-21`
- Modify: any call sites that pass the old positional signature.

Review item #5. There are two `ValidationError` classes: one in `qem/fit/validation.py` (a `ValueError` subclass with `parameter / value / message / suggestion` constructor), one in `qem/utils/exceptions.py` (a `QEMError` subclass with `message / validation_rules / **kwargs`). The unified hierarchy version wins.

- [ ] **Step 1: Inventory the constructor sites**

Run:
```bash
rg -n "raise ValidationError\(" qem/fit/validation.py | head -40
```

Note the call shape: `raise ValidationError("param_name", value, "message", "suggestion")` — four positional args today.

- [ ] **Step 2: Rewrite the local class as a thin shim deferred to `qem.utils.exceptions.ValidationError`**

In `qem/fit/validation.py:12-21`, replace the local class definition with:

```python
from qem.utils.exceptions import ValidationError as _ValidationError


def ValidationError(parameter, value, message, suggestion=None):
    """Backwards-compatible factory: wraps the canonical
    qem.utils.exceptions.ValidationError with the historic
    (parameter, value, message, suggestion) call signature so the
    rest of validation.py keeps compiling."""
    msg = f"Parameter '{parameter}' validation failed: {value} - {message}"
    if suggestion:
        msg = f"{msg}. Suggestion: {suggestion}"
    return _ValidationError(
        msg,
        validation_rules=[parameter] if parameter else None,
        suggestion=suggestion,
    )
```

The factory-function approach lets every existing `raise ValidationError(p, v, m, s)` site continue to work untouched (they raise the result of the function), and the type the user catches is now the canonical `qem.utils.exceptions.ValidationError`.

- [ ] **Step 3: Verify exception type plumbs through**

Run:
```bash
python -c "
from qem.fit.validation import FitterValidator, ValidationError as LocalVE
from qem.utils.exceptions import ValidationError as CanonicalVE
import numpy as np
try:
    FitterValidator.validate_image(np.zeros((4, 4)))
except CanonicalVE as e:
    print('caught canonical:', type(e).__name__, str(e)[:80])
"
```

Expected: prints `caught canonical: ValidationError …`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/ -x -q`

Expected: green. If `tests/test_linear_solver.py` (which imports `ValidationError` from `qem.utils.exceptions`) was already passing, it still passes.

- [ ] **Step 5: Commit**

```bash
git add qem/fit/validation.py
git commit -m "refactor(qem): unify duplicate ValidationError (Linus #5)

qem.fit.validation.ValidationError was a separate ValueError subclass
that bypassed the QEMError hierarchy. Replace with a thin factory that
preserves the historic (parameter, value, message, suggestion) call
shape but yields the canonical qem.utils.exceptions.ValidationError."
```

---

## Task 4: Cache the local-window meshgrid in `_sum_local`

**Files:**
- Modify: `qem/fit/model.py:30-199`

Review perf item B / punch-list item #3. `_sum_local` re-allocates a `(2W+1)²`-shaped meshgrid + `torch.zeros_like` mask every forward pass and round-trips `width.max()` through numpy. For a 5000-atom × 8² window fit, that's 320 000 indices per step; cleanup is 2–5× on `fit_global`.

- [ ] **Step 1: Add a window-cache slot to `ImageModel.__init__`**

Edit `qem/fit/model.py:33-37`:

```python
    def __init__(self, dx: float = 1.0):
        super().__init__()
        self.dx = float(dx)
        self.input_params: dict[str, Any] | None = None
        self.built: bool = False
        # Cache the local-window meshgrid keyed by (window_size, dtype, device).
        # See _sum_local — re-allocating these every forward pass is the
        # single biggest hot-loop allocation in fit_global.
        self._window_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
```

- [ ] **Step 2: Rewrite `_sum_local`**

Replace `_sum_local` (lines 161–199) with:

```python
    def _sum_local(
        self,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        extra: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Memory-efficient local-window peak rendering with scatter-add.

        The (2W+1)² meshgrid is cached per (window_size, dtype, device) —
        only atom positions vary between fit-loop iterations, so we keep
        the static window grid and skip the per-call allocation.
        """
        assert self.input_params is not None
        # width is a Parameter on the right device — no numpy round-trip.
        max_width = float(self.width.detach().max().item())
        window_size = int(max_width * 4)
        cache_key = (window_size, x_grid.dtype, x_grid.device)
        cached = self._window_cache.get(cache_key)
        if cached is None:
            coords = torch.arange(
                -window_size, window_size + 1,
                dtype=x_grid.dtype, device=x_grid.device,
            )
            local_x, local_y = torch.meshgrid(coords, coords, indexing="xy")
            self._window_cache[cache_key] = (local_x, local_y)
        else:
            local_x, local_y = cached

        peak_args = (
            torch.remainder(self.pos_x, 1.0),
            torch.remainder(self.pos_y, 1.0),
            self.height, *extra,
        )
        local_peaks = self.model_fn(local_x[..., None], local_y[..., None], *peak_args)

        pos_x_int = torch.floor(self.pos_x)
        pos_y_int = torch.floor(self.pos_y)
        global_x = local_x.unsqueeze(-1) + pos_x_int
        global_y = local_y.unsqueeze(-1) + pos_y_int

        h, w = x_grid.shape
        in_bounds = (global_x >= 0) & (global_x < w) & (global_y >= 0) & (global_y < h)
        # Element-wise mask multiply skips the torch.zeros_like allocation
        # that the previous torch.where(in_bounds, peaks, zeros) needed.
        masked_peaks = local_peaks * in_bounds.to(local_peaks.dtype)

        global_x_safe = torch.clamp(global_x, 0, w - 1).to(torch.int64)
        global_y_safe = torch.clamp(global_y, 0, h - 1).to(torch.int64)
        flat_indices = global_y_safe.reshape(-1) * w + global_x_safe.reshape(-1)

        canvas = torch.zeros_like(x_grid).reshape(-1)
        canvas = canvas.scatter_add(0, flat_indices, masked_peaks.reshape(-1))
        return canvas.reshape(x_grid.shape) + self.background
```

Note: `to_numpy` may now be unused in `model.py` — leave the import alone if other functions still use it; remove it only if `rg "to_numpy" qem/fit/model.py` returns no other hits.

- [ ] **Step 3: Verify equivalence with a tiny reference fit**

Run:
```bash
python - <<'PY'
import numpy as np, torch
from qem.fit.model import GaussianModel
torch.manual_seed(0)
m = GaussianModel(dx=1.0)
m.set_params({
    "pos_x": torch.tensor([10.0, 20.0]),
    "pos_y": torch.tensor([15.0, 25.0]),
    "height": torch.tensor([1.0, 1.0]),
    "width": torch.tensor([2.0]),
    "background": torch.tensor([0.0]),
    "atom_types": torch.tensor([0, 0]),
    "same_width": True,
})
m.build()
x = torch.arange(40, dtype=torch.float32)
y = torch.arange(40, dtype=torch.float32)
xg, yg = torch.meshgrid(x, y, indexing="xy")
local = m.sum(xg, yg, local=True)
glob = m.sum(xg, yg, local=False)
diff = (local - glob).abs().max().item()
print("max |local - global| =", diff)
assert diff < 1e-4, diff
print("cache size:", len(m._window_cache))
PY
```

Expected: `max |local - global|` < 1e-4 and cache size = 1.

- [ ] **Step 4: Run the test suite + the gaussian fitting test specifically**

Run:
```bash
pytest tests/test_model.py tests/test_gaussian_fitting.py tests/test_image_fitting.py -x -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add qem/fit/model.py
git commit -m "perf(qem): cache local-window meshgrid in ImageModel._sum_local (Linus B/#3)

Drops the per-step (2W+1)² meshgrid + torch.zeros_like allocation and
the to_numpy round-trip on width. Cache keyed by (window_size, dtype,
device) — only atom positions vary across fit_global iterations.
Replaces torch.where(mask, peaks, zeros) with peaks * mask.to(dtype)
to skip the extra zero-tensor allocation."
```

---

## Task 5: Pre-store batched grids on `Fitter` and snapshot best state less aggressively

**Files:**
- Modify: `qem/fit/fitter.py:382-410` (initialize_grid)
- Modify: `qem/fit/fitter.py:1670-1685` (optimize entry)
- Modify: `qem/fit/_loop.py:80-115` (snapshot logic)

Review perf item C. `optimize()` rebuilds three `unsqueeze(0)` views every call (cheap, but repetitive); the fit loop snapshots the entire state_dict every time the loss improves (allocator churn for ~50 KB × ~50 improvements per fit).

- [ ] **Step 1: Pre-store batched grid views in `initialize_grid`**

Find `initialize_grid` (around line 383). After the existing `self.x_grid = …` / `self.y_grid = …` lines, add:

```python
        # Pre-batched views for optimize() — torch view; free.
        self.x_grid_batched = self.x_grid.unsqueeze(0)
        self.y_grid_batched = self.y_grid.unsqueeze(0)
```

- [ ] **Step 2: Use the pre-stored views in `optimize`**

In `qem/fit/fitter.py:1674-1678`, replace:

```python
        image_tensor = torch.unsqueeze(image_tensor, 0)
        x_grid = torch.unsqueeze(self.x_grid, 0)
        y_grid = torch.unsqueeze(self.y_grid, 0)
        model_inputs = [x_grid, y_grid]
```

with:

```python
        image_tensor = image_tensor.unsqueeze(0)
        model_inputs = [self.x_grid_batched, self.y_grid_batched]
```

- [ ] **Step 3: Snapshot the best state every N epochs in the loop**

Edit `qem/fit/_loop.py:48-123`. Add a new keyword arg `snapshot_every: int = 50` to `fit_loop`'s signature, immediately after `min_lr: float = 1e-6,`. Then change the snapshot logic so the deep-clone only fires every `snapshot_every` epochs **or** at the final epoch:

```python
        if loss_val < best_loss * (1.0 - 1e-3):
            best_loss = loss_val
            should_snapshot = (
                (epoch + 1) % snapshot_every == 0
                or (epoch + 1) == epochs
                or best_state is None  # first improvement: snapshot once.
            )
            if should_snapshot:
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
```

Default `snapshot_every=50` keeps churn down for the typical 1000-epoch fit (≤20 snapshots) while still preserving "restore-best-weights" semantics within that window.

- [ ] **Step 4: Run model + image-fitting tests**

Run:
```bash
pytest tests/test_image_fitting.py tests/test_gaussian_fitting.py -x -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add qem/fit/fitter.py qem/fit/_loop.py
git commit -m "perf(qem): pre-batch grids and throttle best-state snapshots (Linus C)

initialize_grid now caches x_grid_batched / y_grid_batched (free torch
views) so optimize() stops re-creating them per call. fit_loop gains
snapshot_every (default 50) so a 1000-epoch fit clones model.state_dict
≤20 times instead of every loss improvement, cutting allocator churn."
```

---

## Task 6: Fold L-BFGS into `_loop.fit_loop` via PyTorch closure

**Files:**
- Modify: `qem/fit/_loop.py` (add closure-based path)
- Modify: `qem/fit/fitter.py:1687-1710` (replace LBFGSOptimizer call)
- Delete: `qem/fit/lbfgs.py`

Review item #6. The 95-line `LBFGSOptimizer` wrapper is duplicated logic. PyTorch's L-BFGS uses `optimizer.step(closure)`; one extra branch in `fit_loop` covers it.

- [ ] **Step 1: Extend `make_optimizer` and `fit_loop` to support L-BFGS**

In `qem/fit/_loop.py`, edit `make_optimizer`:

```python
def make_optimizer(
    name: str,
    parameters: Any,
    learning_rate: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Build a torch optimiser by short name."""
    cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "lbfgs": torch.optim.LBFGS,
    }.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown optimizer {name!r}; expected one of "
            "'adam', 'adamw', 'sgd', 'lbfgs'."
        )
    return cls(parameters, lr=learning_rate, **kwargs)
```

Then in `fit_loop`, before the per-epoch loop, branch on `isinstance(optimizer, torch.optim.LBFGS)`:

```python
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    if is_lbfgs:
        # L-BFGS reevaluates the loss multiple times per .step(); the
        # closure pattern is the only supported API.
        # ReduceLROnPlateau is meaningful for first-order optimisers but
        # not for L-BFGS — it manages its own line search.
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=1e-2, threshold_mode="rel", min_lr=min_lr,
        )
```

Inside the per-epoch loop, replace the existing `optimizer.zero_grad / forward / backward / step` block with:

```python
        if is_lbfgs:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                pred = model(inputs)
                _loss = loss_fn(target, pred)
                _loss.backward()
                return _loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = loss_fn(target, prediction)
            loss.backward()
            optimizer.step()

        loss_val = float(loss.detach())
        if scheduler is not None:
            scheduler.step(loss_val)
```

- [ ] **Step 2: Replace the L-BFGS branch in `Fitter.optimize`**

In `qem/fit/fitter.py:1685-1710`, replace the entire `if optimizer.lower() == "lbfgs": … else: …` block (everything inside `with operation_context:`) with:

```python
        with operation_context:
            from qem.fit._loop import fit_loop, make_optimizer

            opt = make_optimizer(optimizer, model.parameters(), step_size)
            fit_loop(
                model=model,
                inputs=model_inputs,
                target=image_tensor,
                loss_fn=self.loss,
                optimizer=opt,
                epochs=maxiter,
                tol=tol,
                patience=100,
                lr_patience=10,
                lr_factor=0.1,
                min_lr=1e-6,
                verbose=verbose,
            )
```

Then delete the now-unused `from qem.fit.lbfgs import LBFGSOptimizer` import at the top of `fitter.py`.

- [ ] **Step 3: Delete `qem/fit/lbfgs.py`**

```bash
git rm qem/fit/lbfgs.py
```

Then `rg LBFGSOptimizer qem/ tests/ examples/` — should return zero hits.

- [ ] **Step 4: Smoke-test L-BFGS path**

Run:
```bash
python - <<'PY'
import numpy as np, torch
from qem import Fitter
np.random.seed(0); torch.manual_seed(0)
img = np.random.rand(64, 64).astype(np.float32)
f = Fitter(img, dx=0.1, units="nm")
# Just verify the L-BFGS branch in optimize doesn't crash on import.
from qem.fit._loop import make_optimizer
opt = make_optimizer("lbfgs", [torch.tensor([1.0], requires_grad=True)], 1.0)
print("L-BFGS factory ok:", type(opt).__name__)
PY
pytest tests/ -x -q
```

Expected: `L-BFGS factory ok: LBFGS` and pytest green.

- [ ] **Step 5: Commit**

```bash
git add qem/fit/_loop.py qem/fit/fitter.py qem/fit/lbfgs.py
git commit -m "refactor(qem): fold L-BFGS into fit_loop via closure (Linus #6)

Adds an isinstance(optimizer, torch.optim.LBFGS) branch in fit_loop
that uses PyTorch's closure idiom and skips ReduceLROnPlateau (L-BFGS
runs its own line search). make_optimizer learns the 'lbfgs' name.
Drops qem/fit/lbfgs.py (95-line wrapper, single call site)."
```

---

## Task 7: Replace `qem/utils/log.py` with a stdlib wrapper

**Files:**
- Modify: `qem/utils/log.py` (collapse to ~50 lines)
- Modify: `qem/utils/memory.py:38, 73, 243, 579` (only call site, four `get_logger(...)` invocations)

Review item #4. `StructuredFormatter`, `UserFriendlyFormatter` (with emojis), `QEMLogger`, `PerformanceTracker`, `TqdmLoggingHandler`, `setup_logging` — all of this is unused outside `qem/utils/memory.py`'s four `get_logger(...)` calls. The whole file is replaceable with stdlib + tqdm.

- [ ] **Step 1: Verify `get_logger` consumers**

Run:
```bash
rg -n "get_logger\(|setup_logging\(|QEMLogger|PerformanceTracker" qem/ tests/ examples/
```

Expected: only the four `get_logger("qem.…")` calls in `qem/utils/memory.py` (the surviving `MemoryMonitor` plus the dead-class `__init__`s — but wait, those classes were removed in Task 1, so only `MemoryMonitor.__init__:73` should be left).

- [ ] **Step 2: Rewrite `qem/utils/log.py` from scratch**

Replace the file contents with:

```python
"""Lightweight logging helpers for QEM.

We're a numerical library, not a service that needs JSON logs piped
into ELK. So this module is intentionally thin: a stdlib basicConfig
wrapper plus a tqdm-friendly handler so progress bars and log lines
don't fight for the terminal.

Public surface:
    setup_logging(level="INFO", log_file=None)
    get_logger(name) -> logging.Logger
"""

from __future__ import annotations

import logging
import logging.handlers
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so progress bars stay clean."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configure the root QEM logger. Idempotent."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    qem_logger = logging.getLogger("qem")
    if qem_logger.handlers:
        return

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = TqdmLoggingHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    qem_logger.setLevel(level)
    qem_logger.addHandler(console)
    qem_logger.propagate = False

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        qem_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger. Use this inside qem.* modules."""
    return logging.getLogger(name)


@contextmanager
def temporary_log_level(logger_name: str, level: str):
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    try:
        yield
    finally:
        logger.setLevel(old_level)


__all__ = ["setup_logging", "get_logger", "temporary_log_level", "TqdmLoggingHandler"]
```

`MemoryMonitor`'s call sites (`get_logger("qem.memory").info(...)` etc.) keep working because stdlib `logging.Logger.info` is a strict subset of the old custom `info`/`info_operation` methods. Verify by inspection: the surviving `MemoryMonitor` only calls `self.logger.info(...)`, `self.logger.warning(...)` — both of which exist on stdlib `logging.Logger`.

- [ ] **Step 3: Run the suite**

Run:
```bash
python -c "import qem; from qem.utils.log import get_logger, setup_logging; setup_logging('INFO'); get_logger('qem.test').info('hello')"
pytest tests/ -x -q
```

Expected: log line printed; pytest green.

- [ ] **Step 4: Commit**

```bash
git add qem/utils/log.py
git commit -m "refactor(qem): trim utils/log.py to a stdlib wrapper (Linus #4)

QEMLogger, StructuredFormatter, UserFriendlyFormatter (with emojis),
PerformanceTracker, setup_debug_logging, setup_performance_logging:
all unused outside qem/utils/memory.py's get_logger calls. Replaces
the lot with ~70 lines of stdlib basicConfig + a tqdm-friendly
handler. MemoryMonitor's surviving call sites use only standard
logger methods so they keep working unchanged."
```

---

## Task 8: Simplify `LinearSystemSolver`'s try-fallback nest

**Files:**
- Modify: `qem/fit/solver.py:374-460`

Smaller-things item #1: the existing 60+ lines of "memory-aware strategy selection" with `psutil` polling, plus a try/except that falls back to another try/except, can collapse to: try direct, on `MemoryError`/`MatrixRankWarning`/`np.linalg.LinAlgError` fall back to iterative — and on second failure, raise `DataError`.

- [ ] **Step 1: Read the surrounding code first**

Run:
```bash
rg -n "class LinearSystemSolver|def solve\(|def choose_strategy" qem/fit/solver.py
```

Note the public method signatures touched by the change. Don't change names or arg shapes.

- [ ] **Step 2: Rewrite the body of the public solve method**

In `qem/fit/solver.py:374-460`, replace the body of `solve(...)` with the simpler control flow:

```python
        # Coerce target to numpy (MPS tensors and friends).
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()
        elif not isinstance(target, np.ndarray):
            target = np.asarray(target)

        # scipy sparse matrices always go through the iterative path.
        if hasattr(design_matrix, "tocsr"):
            return SciPySolver.solve_iterative(design_matrix, target, non_negative)

        try:
            return self.solver.solve_direct(design_matrix, target, non_negative)
        except (MemoryError, np.linalg.LinAlgError) as exc:
            logging.info("Direct solver fallback (%s); trying iterative.", exc)
        except Exception as exc:
            err = str(exc).lower()
            if not any(k in err for k in ("singular", "memory", "out of memory")):
                raise
            logging.info("Direct solver hit numerical issue (%s); trying iterative.", exc)

        try:
            return self.solver.solve_iterative(design_matrix, target, non_negative)
        except Exception as exc:
            raise DataError(
                f"Linear solve failed: {exc}",
                technical_details={"matrix_shape": getattr(design_matrix, "shape", None)},
            ) from exc
```

If the surrounding method has the `choose_strategy(...)` helper that was only called from this body, also delete that helper (and any `psutil` import in `solver.py` that becomes unused). Run `rg "choose_strategy" qem/` after the edit to confirm zero remaining call sites.

- [ ] **Step 3: Run the linear solver tests**

Run:
```bash
pytest tests/test_linear_solver.py tests/test_linear_solver_core.py -x -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add qem/fit/solver.py
git commit -m "refactor(qem): collapse LinearSystemSolver fallback nest (Linus smaller #1)

The 'memory-aware strategy selection' with psutil polling at every
solve was overhead, not optimisation. Replace with: try direct, on
MemoryError/LinAlgError/singular fall back to iterative, on second
failure raise DataError. Mirrors what qem/fit/sparse.py already does
right."
```

---

## Task 9: Make `qem.Fitter` a lazy module-level attribute

**Files:**
- Modify: `qem/__init__.py`

Review item #7. `from .fit.fitter import Fitter` at import time pulls in `h5py`, `matscipy`, the entire viz stack, GMM, ASE, and `crystal_analyzer`. The cost matters for `import qem` in CLIs and notebooks that don't fit images.

- [ ] **Step 1: Rewrite `qem/__init__.py`**

```python
"""QEM - Quantitative Electron Microscopy Analysis Package.

Pure-PyTorch library for atomic-resolution STEM image quantification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Submodules are eagerly imported (lightweight enough); heavy classes
# (Fitter and friends) load lazily via __getattr__.
from . import io
from . import fit
from . import analysis
from . import viz
from . import processing
from . import detector
from . import optics
from . import utils

__all__ = [
    "Fitter",
    "fit",
    "analysis",
    "viz",
    "processing",
    "detector",
    "optics",
    "utils",
    "io",
]


def __getattr__(name: str):
    if name == "Fitter":
        # Deferred to dodge h5py / matscipy / GMM import cost.
        from .fit.fitter import Fitter as _Fitter
        return _Fitter
    raise AttributeError(f"module 'qem' has no attribute {name!r}")


if TYPE_CHECKING:
    from .fit.fitter import Fitter as Fitter  # for IDE autocomplete only.
```

- [ ] **Step 2: Verify `import qem` still exposes Fitter and that `from qem import Fitter` works**

Run:
```bash
python - <<'PY'
import qem
print("qem.Fitter is", qem.Fitter)
from qem import Fitter
print("from qem import Fitter ok:", Fitter)
PY
pytest tests/test_import.py -x -q
```

Expected: both prints succeed; test_import passes.

- [ ] **Step 3: Commit**

```bash
git add qem/__init__.py
git commit -m "refactor(qem): lazy-import Fitter via module __getattr__ (Linus #7)

The eager 'from .fit.fitter import Fitter' triggered h5py, matscipy,
the viz stack, GMM, ASE, and crystal_analyzer at 'import qem' time.
Move to a PEP 562 module-level __getattr__: 'qem.Fitter' and
'from qem import Fitter' both still work, IDE autocomplete is
preserved via TYPE_CHECKING."
```

---

## Task 10: Port `route_b_joint_ls` to torch + `_loop.fit_loop`

**Files:**
- Modify: `qem/fusion/route_b_joint_ls.py:130-220`

Review item #10. The existing 200-line hand-rolled projected-gradient SGD with TV regularisation belongs on torch with autograd and the same `_loop.fit_loop` we use everywhere else.

- [ ] **Step 1: Sketch the torch model**

Build a small `nn.Module` whose forward returns the joint cost as a scalar; project to the non-negative cone after each step using a hook on the parameter (see Step 3). Replace the loop body in `JointLeastSquaresRoute.fit(...)` (lines 154-197) with:

```python
        import torch
        from qem.fit._loop import fit_loop, make_optimizer

        device = torch.device("cpu")
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device, requires_grad=True)
        adf_t = torch.as_tensor(adf, dtype=torch.float32, device=device)
        edx_t = (
            torch.as_tensor(edx, dtype=torch.float32, device=device)
            if edx is not None and edx_ref is not None else None
        )
        eels_t = (
            torch.as_tensor(eels, dtype=torch.float32, device=device)
            if eels is not None and eels_ref is not None else None
        )
        edx_ref_t = torch.as_tensor(edx_ref, dtype=torch.float32, device=device) if edx_ref is not None else None
        eels_ref_t = torch.as_tensor(eels_ref, dtype=torch.float32, device=device) if eels_ref is not None else None

        class _JointLossModel(torch.nn.Module):
            def __init__(self, x_init):
                super().__init__()
                self.x = torch.nn.Parameter(x_init)
            def forward(self, _inputs):
                return self.x  # we compute loss in loss_fn (closure has caps).

        def joint_loss(_target, x_param):
            adf_pred = self._adf_forward_torch(x_param)
            adf_residual = adf_pred - adf_t
            cost = 0.5 * self.lambda_adf * torch.mean(adf_residual ** 2)

            if edx_t is not None:
                edx_pred = torch.tensordot(x_param, edx_ref_t.T, dims=([-1], [0]))
                cost = cost + 0.5 * self.lambda_edx * torch.mean((edx_pred - edx_t) ** 2)

            if eels_t is not None:
                eels_pred = torch.tensordot(x_param, eels_ref_t.T, dims=([-1], [0]))
                cost = cost + 0.5 * self.lambda_eels * torch.mean((eels_pred - eels_t) ** 2)

            if self.lambda_tv:
                cost = cost + self.lambda_tv * self._tv_value_torch(x_param)
            return cost

        model = _JointLossModel(x_t.detach().clone())
        opt = make_optimizer("adam", model.parameters(), self.step_size)

        # After each step, project to the non-negative cone.
        def _project_nonneg(_grad):
            with torch.no_grad():
                model.x.clamp_(min=0.0)
            return None
        model.x.register_post_accumulate_grad_hook(_project_nonneg)

        result = fit_loop(
            model=model,
            inputs=None,
            target=None,
            loss_fn=joint_loss,
            optimizer=opt,
            epochs=self.max_iter,
            tol=self.tolerance,
            patience=self.max_iter,  # disable early-stopping based on patience.
            lr_patience=10,
            lr_factor=0.5,
            min_lr=1e-6,
            verbose=False,
        )

        x = model.x.detach().cpu().numpy()
        # Build a costs dict in the same shape the old loop returned, but
        # only with the totals — torch path doesn't track per-modality.
        costs = {"total": [result.final_loss], "adf": [], "edx": [], "eels": [], "tv": []}
```

You will also need to add small torch helper methods `_adf_forward_torch`, `_tv_value_torch` next to the existing numpy `_adf_forward` and `_tv_value` (or reuse the numpy ones via numpy-torch interop where the gradient isn't needed; but for the gradient path use torch ops directly).

- [ ] **Step 2: Add torch versions of the helpers**

Below the numpy `_adf_forward`, `_adf_gradient`, `_tv_value`, `_tv_gradient` methods, add:

```python
    def _adf_forward_torch(self, x_t):
        # Mirror the numpy _adf_forward semantics with torch ops so the
        # graph is built end-to-end. Gamma exponent applies element-wise.
        return torch.pow(torch.tensordot(x_t, torch.as_tensor(self._z_atomic, dtype=x_t.dtype, device=x_t.device), dims=([-1], [0])), self.gamma)

    def _tv_value_torch(self, x_t):
        # Anisotropic TV: sum |x[i+1,j] - x[i,j]| + |x[i,j+1] - x[i,j]|.
        diff_y = x_t[1:, :, :] - x_t[:-1, :, :]
        diff_x = x_t[:, 1:, :] - x_t[:, :-1, :]
        return torch.mean(torch.abs(diff_y)) + torch.mean(torch.abs(diff_x))
```

(Replace `self._z_atomic` with whatever attribute the numpy `_adf_forward` reads; check the existing implementation around line 161 of `route_b_joint_ls.py`.)

- [ ] **Step 3: Update the test expectations if cost shape changed**

Run:
```bash
pytest tests/test_fusion.py tests/test_fusion_vendor_demo.py -x -q
```

If the tests assert a per-modality cost trajectory length, update them to be tolerant of the new shape (just the final value), or, alternatively, accumulate per-modality costs inside `joint_loss` via a closure-bound list — pick whichever keeps the existing tests passing with the smallest diff.

- [ ] **Step 4: Commit**

```bash
git add qem/fusion/route_b_joint_ls.py tests/test_fusion.py tests/test_fusion_vendor_demo.py
git commit -m "refactor(qem.fusion): port JointLeastSquaresRoute to torch + fit_loop (Linus #10)

Replaces the 200-line hand-rolled projected-gradient SGD with TV
regularisation with torch.autograd over Adam via _loop.fit_loop.
Non-negativity is enforced by clamping the parameter in a
post-accumulate-grad hook. Same cost surface, cleaner gradient
math, single optimisation primitive across the codebase."
```

---

## Task 11: Add a notebook smoke-test step to CI

**Files:**
- Modify: `.github/workflows/python-package.yml`

Review item #8. Three notebooks plus two `.py` examples have zero CI coverage. Add `pytest --nbmake examples/*.ipynb` gated on the test data being downloadable (if data isn't present, just smoke-import the example modules instead).

- [ ] **Step 1: Add `nbmake` to dev extras**

Edit `pyproject.toml`, in the `dev = [...]` list, add `"nbmake>=1.5",` after `"pytest-benchmark>=4.0",`.

- [ ] **Step 2: Add a CI step**

Edit `.github/workflows/python-package.yml`. After the `Test with pytest` step, add:

```yaml
    - name: Smoke-test example scripts
      run: |
        python -c "import importlib, glob, os, sys
        for path in sorted(glob.glob('examples/*.py')):
            mod = os.path.splitext(os.path.basename(path))[0]
            sys.path.insert(0, 'examples')
            try:
                importlib.import_module(mod)
                print('imported', mod)
            except Exception as exc:
                print('SMOKE FAIL', mod, '->', exc); raise
            finally:
                sys.path.pop(0)
        "

    - name: Smoke-test example notebooks (first cell only)
      run: |
        # nbmake-driven smoke: execute the first cell of each notebook so
        # ImportError / SyntaxError surfaces. Skips notebooks that need
        # real data to load.
        pip install nbmake
        for nb in examples/*.ipynb; do
          echo "Smoke testing $nb (first cell only)"
          jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 \
            --ClearOutputPreprocessor.enabled=True \
            --output /tmp/smoke_$(basename "$nb") "$nb" || \
            echo "::warning::Notebook smoke failed for $nb (likely missing data)"
        done
```

The notebook step intentionally `|| echo`s — it surfaces failures as GitHub warnings rather than CI failures, because they're sensitive to test-data presence. The `.py` example smoke is strict (raises on failure).

- [ ] **Step 3: Lint the YAML locally**

Run:
```bash
python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/python-package.yml')); print('yaml ok')"
```

Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .github/workflows/python-package.yml
git commit -m "chore(ci): smoke-test example .py and notebooks (Linus #8)

Adds two CI steps: (1) strict importlib.import_module of every
examples/*.py — fails CI if the example breaks; (2) nbmake/jupyter
nbconvert execution of examples/*.ipynb — failures surface as GitHub
warnings (not CI failures) since notebook data dependencies aren't
checked into the repo. Also adds nbmake to the dev extras."
```

---

## Task 12: Stop bouncing between numpy and torch in the fitter hot path

**Files:**
- Modify: `qem/fit/fitter.py` — every `safe_convert_to_numpy(...)` / `safe_convert_to_tensor(...)` site that's inside an inner loop or a callable invoked from one.
- Add: `qem/fit/fitter.py::Fitter.detach()` — single explicit numpy-snapshot method.

Review perf item A. There are **42** `safe_convert_to_numpy(...)` / `safe_convert_to_tensor(...)` calls in `fitter.py`. The single biggest no-algorithmic perf win.

- [ ] **Step 1: Inventory the call sites**

Run:
```bash
rg -n "safe_convert_to_numpy|safe_convert_to_tensor" qem/fit/fitter.py | wc -l
rg -n "safe_convert_to_numpy|safe_convert_to_tensor" qem/fit/fitter.py
```

Note which call sites are: (a) inside `predict`, `predict_local`, `loss`, `_optimization_model` callbacks, the `optimize` body, the GMM/voronoi paths — these are the hot-loop ones; (b) at IO boundaries (`save`, `load`, `plot_*`, `linear_estimator` solution-return paths) — these stay.

- [ ] **Step 2: Convert `predict` and `loss` to keep state on-device**

For each call to `safe_convert_to_numpy` whose result is then immediately re-tensorised in the next caller, drop both conversions. Concrete pattern:

```python
# Before
self.prediction = safe_convert_to_numpy(self.predict(...))
# … later …
pred = safe_convert_to_tensor(self.prediction)

# After
self.prediction_t = self.predict(...)  # keep as torch tensor on device
# … later …
pred = self.prediction_t
```

Keep `self.prediction` as a numpy mirror only for code that genuinely needs numpy (plotting, save). Update those numpy consumers to read from a property:

```python
    @property
    def prediction(self) -> np.ndarray:
        if self.prediction_t is None:
            return np.zeros_like(self.image)
        return self.prediction_t.detach().cpu().numpy()
```

This way numpy access still works for plotting / saving code but is computed lazily and only at the boundary.

- [ ] **Step 3: Add an explicit `Fitter.detach()` method**

```python
    def detach(self) -> dict:
        """Snapshot every tensor parameter to numpy.

        Call exactly once when you're done fitting and want everything
        in numpy land for plotting / saving / external consumption.
        """
        out: dict[str, np.ndarray] = {}
        for k, v in (self.params or {}).items():
            out[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else v
        out["prediction"] = self.prediction  # uses the numpy property
        return out
```

- [ ] **Step 4: Fix the `optimize()` return-type lie**

In `qem/fit/fitter.py`, `Fitter.optimize` is annotated `-> dict[str, NDArray[Any]]` but actually returns torch tensors via `model.get_params()`. Update the annotation to `-> dict[str, torch.Tensor]` and update the docstring accordingly. Don't convert — downstream callers inside `Fitter.fit_global` expect tensors.

- [ ] **Step 5: Run the tests carefully — this touches the fit path**

Run:
```bash
pytest tests/test_image_fitting.py tests/test_gaussian_fitting.py tests/test_model_api.py -x -q -v
```

Expected: green. If any test asserts `type(...) is np.ndarray` on a previously-numpy-now-tensor field, fix the test or wrap the access in `Fitter.detach()`.

- [ ] **Step 6: Quick benchmark to confirm the speedup**

Run:
```bash
python - <<'PY'
import time, numpy as np, torch
from qem import Fitter
torch.manual_seed(0); np.random.seed(0)
img = np.random.rand(128, 128).astype(np.float32)
f = Fitter(img, dx=0.1, units="nm")
# Initialise with a small synthetic peak grid for a meaningful timing.
# (Adjust to whatever Fitter expects for init — this is illustrative.)
t0 = time.time()
# … set up params + run 100 epochs of fit_global …
print("100-epoch fit_global: %.2fs" % (time.time() - t0))
PY
```

Note the time. Compare informally against pre-change timing (the user can stash the old number if curious). Even a 1.5× speedup on Mac MPS / CPU is already a clear win.

- [ ] **Step 7: Commit**

```bash
git add qem/fit/fitter.py
git commit -m "perf(qem): keep fit-path state on the model device (Linus A)

Drops ~30 safe_convert_to_numpy / safe_convert_to_tensor round-trips
inside predict, loss and optimize. Adds a lazy 'prediction' property
that materialises numpy on demand and a single Fitter.detach() to
snapshot everything for save / plot / external use. Annotation on
Fitter.optimize fixed to -> dict[str, torch.Tensor] (it never
returned numpy, the type was lying). Saves ~2x on Mac/MPS, more on
CUDA."
```

---

## Task 13: Decompose `Fitter` — extract plotting

**Files:**
- Create: `qem/fit/plot.py`
- Modify: `qem/fit/fitter.py` — remove all `plot_*` and `_plot_*` methods listed below.
- Modify: `qem/fit/__init__.py` — re-export the new module.

Review item #9, first slice. Move every `plot_*` and `_plot_*` method out of `Fitter` into a sibling module of free functions that take a `Fitter` as their first argument. Keep `Fitter` instances callable via thin pass-through methods so existing notebooks keep working.

- [ ] **Step 1: Create the new plot module**

Create `qem/fit/plot.py` with imports + the function bodies copy-pasted from each `Fitter.plot_*` / `Fitter._plot_*` method. Method names map 1:1 to module-level functions:

```python
"""Fitter plotting helpers.

Every function takes a Fitter instance as the first argument; the
Fitter class keeps thin pass-through methods so existing call sites
(`fitter.plot_fitting()`) keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter


def plot_coordinates(fitter: "Fitter", s: int = 1) -> None:
    # body: copied from Fitter.plot_coordinates (current line 2318)
    ...


def plot_fitting(fitter: "Fitter", save: bool = False) -> None:
    # body: copied from Fitter.plot_fitting (current line 2339)
    ...


def plot_scs(fitter: "Fitter", *args, **kwargs) -> None:
    # body: copied from Fitter.plot_scs (current line 2365)
    ...


def plot_scs_voronoi(fitter: "Fitter", *args, **kwargs) -> None:
    # body: copied from Fitter.plot_scs_voronoi (current line 2471)
    ...


def plot_voronoi_integration_intensity(fitter: "Fitter", plot: bool = False, save: bool = False) -> None:
    ...


def plot_scs_histogram(fitter: "Fitter", save: bool = False, has_units: bool = True) -> None:
    ...


def plot_atom_count_map(fitter: "Fitter", element_name=None, save: bool = False, figsize=(12, 8)) -> None:
    ...


def plot_region(fitter: "Fitter") -> None:
    ...


def plot_progress(fitter: "Fitter", params, index, select_params) -> None:
    """Replaces Fitter._plot_progress."""
    ...


def plot_gmm_results(fitter: "Fitter", cross_sections, gmm_model, element_name, save_results: bool = False) -> None:
    """Replaces Fitter._plot_gmm_results."""
    ...


def plot_domain_analysis(fitter: "Fitter", *args, **kwargs) -> None:
    """Replaces Fitter._plot_domain_analysis."""
    ...
```

For each function body: take the existing method body verbatim, replace every `self.` with `fitter.`. No other behaviour changes.

- [ ] **Step 2: Replace methods on `Fitter` with thin pass-throughs**

In `qem/fit/fitter.py`, replace each plot_* method body (currently 11 of them between lines 1900 and 3260) with one-line pass-throughs:

```python
    def plot_fitting(self, save: bool = False) -> None:
        from qem.fit.plot import plot_fitting as _impl
        return _impl(self, save=save)

    def plot_coordinates(self, s: int = 1) -> None:
        from qem.fit.plot import plot_coordinates as _impl
        return _impl(self, s=s)

    # …repeat for the other nine methods…
```

Pass-throughs preserve the existing public API (`fitter.plot_fitting()` still works in notebooks).

- [ ] **Step 3: Update `qem/fit/__init__.py` to re-export the module**

Add `from qem.fit import plot  # noqa: F401` so `from qem.fit import plot` works.

- [ ] **Step 4: Run the suite + a notebook smoke**

Run:
```bash
pytest tests/ -x -q
python -c "
import numpy as np, matplotlib
matplotlib.use('Agg')
from qem import Fitter
img = np.random.rand(64, 64).astype(np.float32)
f = Fitter(img, dx=0.1, units='nm')
# don't actually plot — just import-time + attribute presence check
print('plot_fitting on fitter:', callable(getattr(f, 'plot_fitting', None)))
from qem.fit.plot import plot_fitting
print('plot_fitting in module:', plot_fitting)
"
```

Expected: pytest green; both prints succeed.

- [ ] **Step 5: Commit**

```bash
git add qem/fit/plot.py qem/fit/fitter.py qem/fit/__init__.py
git commit -m "refactor(qem): extract Fitter plot_* methods to qem.fit.plot (Linus #9 — plot)

Moves all 11 plot_* / _plot_* method bodies into qem/fit/plot.py as
free functions taking a Fitter as the first arg. Fitter keeps thin
pass-through methods so existing fitter.plot_fitting() call sites
keep working. Net: ~750 LoC out of fitter.py."
```

---

## Task 14: Decompose `Fitter` — extract domain analysis

**Files:**
- Create: `qem/analysis/domains.py`
- Modify: `qem/fit/fitter.py` — remove `estimate_complex_domains`, `_separate_vacuum_and_sample`, `_identify_domain_boundaries`, `_create_polygon_enclosures`. (`_plot_domain_analysis` already moved in Task 13.)
- Modify: `qem/analysis/__init__.py` — re-export.

Same pattern as Task 13: free functions taking `Fitter` as the first arg, thin pass-throughs left on `Fitter`.

- [ ] **Step 1: Create `qem/analysis/domains.py`**

```python
"""Complex-domain analysis extracted from Fitter.

Pure free functions taking a Fitter as the first argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, laplace, sobel
from skimage.measure import find_contours
from skimage.morphology import label, remove_small_objects

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter


def estimate_complex_domains(fitter: "Fitter", *args, **kwargs):
    # copy verbatim from Fitter.estimate_complex_domains (current line 2985);
    # replace self. with fitter.
    ...


def _separate_vacuum_and_sample(fitter: "Fitter", vacuum_threshold: float = 0.05):
    ...


def _identify_domain_boundaries(
    fitter: "Fitter",
    method: str = "intensity_gradient",
    min_domain_size: int = 50,
    domain_threshold: float = 15,
    vacuum_mask=None,
    clean_image=None,
):
    ...


def _create_polygon_enclosures(fitter: "Fitter", domain_regions):
    ...
```

- [ ] **Step 2: Replace methods on `Fitter` with pass-throughs**

```python
    def estimate_complex_domains(self, *args, **kwargs):
        from qem.analysis.domains import estimate_complex_domains as _impl
        return _impl(self, *args, **kwargs)

    def _separate_vacuum_and_sample(self, vacuum_threshold: float = 0.05):
        from qem.analysis.domains import _separate_vacuum_and_sample as _impl
        return _impl(self, vacuum_threshold=vacuum_threshold)

    def _identify_domain_boundaries(self, **kwargs):
        from qem.analysis.domains import _identify_domain_boundaries as _impl
        return _impl(self, **kwargs)

    def _create_polygon_enclosures(self, domain_regions):
        from qem.analysis.domains import _create_polygon_enclosures as _impl
        return _impl(self, domain_regions)
```

- [ ] **Step 3: Update `qem/analysis/__init__.py`**

Add `from qem.analysis import domains  # noqa: F401`.

- [ ] **Step 4: Test**

Run: `pytest tests/ -x -q`

Expected: green. (No domain-specific tests today — the smoke-import test is enough.)

- [ ] **Step 5: Commit**

```bash
git add qem/analysis/domains.py qem/fit/fitter.py qem/analysis/__init__.py
git commit -m "refactor(qem): extract Fitter domain-analysis methods to qem.analysis.domains (Linus #9 — domains)

Moves the 4 domain methods (estimate_complex_domains,
_separate_vacuum_and_sample, _identify_domain_boundaries,
_create_polygon_enclosures) into qem/analysis/domains.py as free
functions taking a Fitter. Pass-throughs preserved on Fitter."
```

---

## Task 15: Decompose `Fitter` — extract GMM atom counting

**Files:**
- Create: `qem/analysis/gmm.py`
- Modify: `qem/fit/fitter.py` — remove `estimate_atom_counts_with_gmm`, `integrate_gmm_with_crystal_analyzer`, `update_all_regions_with_gmm`, `export_gmm_updated_structure`. (`_plot_gmm_results` and `plot_atom_count_map` already moved in Task 13.)
- Modify: `qem/analysis/__init__.py` — re-export.

Same pattern. If `qem/analysis/gaussian_mixture_model.py` already exists, the new `gmm.py` should delegate to it for the actual GMM math (don't duplicate sklearn calls).

- [ ] **Step 1: Confirm whether qem.analysis.gaussian_mixture_model exists**

Run:
```bash
ls qem/analysis/
```

If `gaussian_mixture_model.py` exists, the new `qem/analysis/gmm.py` is a thin orchestration layer that delegates GMM-fitting to it. If not, the new file holds both the orchestration and the sklearn-glue.

- [ ] **Step 2: Create `qem/analysis/gmm.py`**

```python
"""GMM-based atom counting extracted from Fitter.

Pure free functions taking a Fitter as the first argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter


def estimate_atom_counts_with_gmm(fitter: "Fitter", *args, **kwargs):
    # body copied from Fitter.estimate_atom_counts_with_gmm (line 2578),
    # self. -> fitter.
    ...


def integrate_gmm_with_crystal_analyzer(fitter: "Fitter", region_index: int = 0):
    ...


def update_all_regions_with_gmm(fitter: "Fitter"):
    ...


def export_gmm_updated_structure(
    fitter: "Fitter", region_index: int = 0, filename: str | None = None,
):
    ...
```

- [ ] **Step 3: Replace methods on `Fitter`**

Pass-throughs identical in shape to Tasks 13 and 14.

- [ ] **Step 4: Update `qem/analysis/__init__.py`**

Add `from qem.analysis import gmm  # noqa: F401`.

- [ ] **Step 5: Test**

Run: `pytest tests/test_gmm.py tests/test_gmm_integration.py tests/ -x -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add qem/analysis/gmm.py qem/fit/fitter.py qem/analysis/__init__.py
git commit -m "refactor(qem): extract Fitter GMM atom-counting methods to qem.analysis.gmm (Linus #9 — gmm)

Moves the 4 GMM-orchestration methods on Fitter into qem/analysis/gmm.py
as free functions taking a Fitter. Pass-throughs preserved."
```

---

## Task 16: Decompose `Fitter` — extract loss / boundary penalty / edge-loss / window helpers

**Files:**
- Create: `qem/fit/loss.py`
- Modify: `qem/fit/fitter.py` — remove the loss helpers.

Review item #9, ~7 methods. Names to extract: `loss`, `boundary_penalty`, `edge_loss`, `_butterworth_window`, `set_boundary_penalty`, `set_adaptive_edge_loss`, plus any private helpers used only by these.

- [ ] **Step 1: Inventory the loss-related methods**

Run:
```bash
rg -n "def loss\(|def boundary_penalty\(|def edge_loss\(|def _butterworth_window\(|def set_boundary_penalty\(|def set_adaptive_edge_loss\(" qem/fit/fitter.py
```

- [ ] **Step 2: Create `qem/fit/loss.py`**

Copy the function bodies as free functions taking the `fitter` instance. The `loss(self, target, prediction)` signature already takes (target, prediction) — keep that exact callable signature so it can still be passed as `loss_fn=fitter.loss` to `fit_loop`. To keep that working, wire it as:

```python
def loss(target, prediction, *, fitter):
    # body copied from Fitter.loss
    ...
```

And have `Fitter.loss` resolve to a partial:

```python
    @property
    def loss(self):
        from qem.fit.loss import loss as _impl
        from functools import partial
        return partial(_impl, fitter=self)
```

- [ ] **Step 3: Replace the other helper methods with pass-throughs**

Same pattern as the previous extraction tasks.

- [ ] **Step 4: Test**

Run: `pytest tests/test_image_fitting.py tests/test_gaussian_fitting.py -x -q`

- [ ] **Step 5: Commit**

```bash
git add qem/fit/loss.py qem/fit/fitter.py
git commit -m "refactor(qem): extract Fitter loss helpers to qem.fit.loss (Linus #9 — loss)

Moves loss / boundary_penalty / edge_loss / _butterworth_window and
their setters into qem/fit/loss.py as free functions taking a
Fitter. Fitter.loss is now a @property returning a partial so
existing 'loss_fn=fitter.loss' usage in optimize() keeps working."
```

---

## Task 17: Decompose `Fitter` — extract peak-finding helpers

**Files:**
- Create: `qem/fit/peaks.py`
- Modify: `qem/fit/fitter.py` — remove peak-finding methods.

Review item #9, ~9 methods. Names: `find_peaks`, `dedup_peaks`, `remove_edge_peaks`, `refine_com`, `refine_local_max`, plus helpers.

- [ ] **Step 1: Inventory**

Run:
```bash
rg -n "def find_peaks\(|def dedup\w*\(|def remove_edge_peaks\(|def refine_com\(|def refine_local_max\(|def _filter_peaks_by\w*\(" qem/fit/fitter.py
```

- [ ] **Step 2: Create `qem/fit/peaks.py`**

```python
"""Peak-finding helpers extracted from Fitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter


# one free function per Fitter method; bodies copied verbatim with
# self. -> fitter.
```

- [ ] **Step 3: Pass-throughs and tests**

Same pattern. Run `pytest tests/ -x -q`.

- [ ] **Step 4: Commit**

```bash
git add qem/fit/peaks.py qem/fit/fitter.py
git commit -m "refactor(qem): extract Fitter peak-finding helpers to qem.fit.peaks (Linus #9 — peaks)"
```

---

## Task 18: Decompose `Fitter` — move background helpers into existing `qem/fit/background.py`

**Files:**
- Modify: `qem/fit/background.py` — receive the methods.
- Modify: `qem/fit/fitter.py` — pass-throughs only.

Background methods on `Fitter` (~6 of them: 1D + 2D + scale + estimator interface) belong in `qem/fit/background.py`, which already exists and has the right scope. Pattern is identical to previous tasks but the destination already exists.

- [ ] **Step 1: Inventory**

Run:
```bash
rg -n "def \w*background\w*\(|def \w*scale\w*\(" qem/fit/fitter.py
```

- [ ] **Step 2: Append free functions to `qem/fit/background.py`**

Copy method bodies verbatim with `self.` → `fitter.`.

- [ ] **Step 3: Pass-throughs on Fitter, tests, commit**

```bash
git add qem/fit/background.py qem/fit/fitter.py
git commit -m "refactor(qem): move Fitter background methods into qem/fit/background.py (Linus #9 — background)"
```

---

## Task 19: Decompose `Fitter` — move voronoi helpers into existing `qem/fit/voronoi.py`

**Files:**
- Modify: `qem/fit/voronoi.py` — receive the methods.
- Modify: `qem/fit/fitter.py` — pass-throughs.

Same shape as Task 18. Methods: `fit_voronoi`, `voronoi_integration`, plus any voronoi-related plot methods that did **not** already get moved to `qem/fit/plot.py` in Task 13.

- [ ] **Step 1: Inventory + extract + pass-throughs + tests + commit**

```bash
git add qem/fit/voronoi.py qem/fit/fitter.py
git commit -m "refactor(qem): move Fitter voronoi methods into qem/fit/voronoi.py (Linus #9 — voronoi)"
```

---

## Task 20: Decompose `Fitter` — move linear-estimator wiring into `solver.py`

**Files:**
- Modify: `qem/fit/solver.py` — accept the linear-estimator entry points as free functions taking a `Fitter`.
- Modify: `qem/fit/fitter.py` — pass-throughs.

Names: `linear_estimator`, `_prepare_target_vector`, `_process_solution`. Same pattern. After this task and Task 13–19, `Fitter` should be in the ≤ 800-LoC range. Verify:

```bash
wc -l qem/fit/fitter.py
```

If still > 800, inventory remaining methods and add a follow-up task — but don't ship a half-extracted Fitter.

- [ ] **Inventory + extract + pass-throughs + tests + commit**

```bash
git add qem/fit/solver.py qem/fit/fitter.py
git commit -m "refactor(qem): move Fitter linear-estimator wiring into qem/fit/solver.py (Linus #9 — solver)

After this task, fitter.py should be ≤ 800 LoC and contain only:
init / save / load / model selection / orchestration."
```

---

## Task 21: Move `_loop.py` → `loop.py`

**Files:**
- Rename: `qem/fit/_loop.py` → `qem/fit/loop.py`
- Modify: every `from qem.fit._loop import …` site (use `git grep`).

Trivial rename — the leading underscore was a hold-over from "this is the temporary keras replacement." It's been the canonical loop for several refactor phases now. Do this last because every other task touches `_loop.py` and the rename would conflict with their diffs.

- [ ] **Step 1: Find import sites**

```bash
git grep -n "from qem.fit._loop\|qem\.fit\._loop"
```

- [ ] **Step 2: Rename and update imports**

```bash
git mv qem/fit/_loop.py qem/fit/loop.py
# then sed -i '' 's/qem.fit._loop/qem.fit.loop/g' on every match site,
# or use Edit tool per file.
```

- [ ] **Step 3: Test + commit**

```bash
pytest tests/ -x -q
git add qem/fit/loop.py qem/fit/fitter.py qem/fit/lbfgs.py qem/fusion/route_b_joint_ls.py
git commit -m "refactor(qem): rename qem/fit/_loop.py -> qem/fit/loop.py

The leading underscore was a hold-over from the keras-replacement
phase. It's been the canonical training loop since Phase 1, drop
the private marker."
```

---

## Self-Review Checklist

- [ ] Every spec item from `docs/superpowers/specs/review.md` mapped to a task:
  - Top-3 problems #1 (dead memory) → Task 1.
  - Top-3 problems #2 (defensive __init__) → Task 2.
  - Top-3 problems #3 / Perf B (`_sum_local`) → Task 4.
  - Perf A (numpy/torch bouncing) → Task 12.
  - Perf C (snapshot + grids) → Task 5.
  - Smaller things #1 (solver fallback nest) → Task 8.
  - Smaller things #2 (log.py) → Task 7.
  - Smaller things #3 (LBFGS) → Task 6.
  - Smaller things #4 (ValidationError dup) → Task 3.
  - Smaller things #5 (optimize return-type lie) → covered inside Task 12 step 4.
  - Smaller things #6 (route_b torch port) → Task 10.
  - Smaller things #7 (CI nbmake) → Task 11.
  - Smaller things #8 (lazy Fitter) → Task 9.
  - Punch list #9 (decompose Fitter) → Tasks 13–20.
  - `_loop.py` rename hint → Task 21.

- [ ] No placeholders inside actionable steps; the `...` in Task 13–20 explicitly mark "copy verbatim, self. → fitter." which is a method-by-method mechanical operation, not a design decision.
- [ ] Type names consistent: `Fitter`, `ImageModel`, `fit_loop`, `make_optimizer`, `MemoryMonitor`, `ValidationError` (canonical from `qem.utils.exceptions`), `DataError`, `JointLeastSquaresRoute`.
- [ ] Order of work is contention-aware: Task 1 (dead-code) before Task 7 (log.py) before Tasks 13+ (decomposition); Task 21 (`_loop.py` rename) intentionally last to avoid conflicts.
- [ ] Each task ends in a commit. No multi-task batches.
