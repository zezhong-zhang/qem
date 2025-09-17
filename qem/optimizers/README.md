# QEM Optimizers Module

This module provides advanced optimization algorithms for image fitting in QEM.

## L-BFGS Optimizer

The L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer is a quasi-Newton method that can converge faster than first-order methods for well-conditioned optimization problems.

### Features

- **PyTorch Backend**: Native L-BFGS implementation using PyTorch's optimizer
- **Unified API**: Same interface regardless of backend
- **Memory Efficient**: Limited-memory approach suitable for large parameter spaces
- **Convergence Monitoring**: Built-in convergence criteria and history tracking

### Usage

#### Basic Usage

```python
from qem.optimizers.lbfgs import LBFGSOptimizer

# Create optimizer
optimizer = LBFGSOptimizer(
    learning_rate=1.0,
    max_iter=20,
    tolerance_grad=1e-7,
)

# Use with model
results = optimizer.optimize(
    model=your_model,
    loss_fn=your_loss_function,  # Required for torch backend
    inputs=model_inputs,
    targets=target_values,
    num_steps=3,
    verbose=True
)
```

#### Integration with ImageFitting

```python
from qem.fit.image_fitting import ImageFitting

# Create ImageFitting instance
fitter = ImageFitting(image, dx=0.1, model_type="gaussian")

# Find peaks and initialize parameters
coordinates = fitter.find_peaks()
params = fitter.init_params()

# Use L-BFGS optimization through unified interface
optimized_params = fitter.fit_global(
    optimizer_type="lbfgs",
    maxiter=200,
    tolerance_grad=1e-6,
    verbose=True
)

# Or use with stochastic fitting
stoch_params = fitter.fit_stochastic(
    optimizer_type="lbfgs",
    batch_size=100,
    maxiter=10
)
```

### Parameters

- `learning_rate`: Step size multiplier (default: 1.0)
- `max_iter`: Maximum iterations per optimization step (default: 20)
- `max_eval`: Maximum function evaluations (default: max_iter * 5/4)
- `tolerance_grad`: Gradient tolerance for convergence (default: 1e-7)
- `tolerance_change`: Parameter change tolerance (default: 1e-9)
- `history_size`: Number of previous gradients to store (default: 100)
- `line_search_fn`: Line search function ('strong_wolfe' or None)
- `backend`: Backend to use ('torch' or 'keras')

### Requirements

- **PyTorch**: Required for true L-BFGS optimization with torch backend
- **Keras**: Required for keras backend (uses AdamW as approximation)

### Performance Notes

- L-BFGS typically converges faster than first-order methods for smooth, well-conditioned problems
- Memory usage scales with `history_size` parameter
- PyTorch backend provides true L-BFGS implementation
- Keras backend uses second-order approximation (AdamW)

### Examples

See `examples/lbfgs_optimization_example.py` for a complete example comparing L-BFGS with standard optimization methods.