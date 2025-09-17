"""L-BFGS Optimizer for PyTorch models."""

import logging
from typing import Dict, Any, Optional, Callable
import numpy as np

try:
    import torch
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LBFGSOptimizer:
    """L-BFGS optimizer for PyTorch models."""
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        maxiter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for L-BFGS")
            
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.max_eval = max_eval or maxiter * 5 // 4
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self._loss_history = []
        self._grad_norm_history = []
    
    def _create_torch_optimizer(self, parameters):
        """Create PyTorch L-BFGS optimizer."""
        return torch_optim.LBFGS(
            parameters, lr=self.learning_rate, max_iter=self.maxiter,
            max_eval=self.max_eval, tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change, history_size=self.history_size,
            line_search_fn=self.line_search_fn
        )
    
    def optimize(self, model, loss_fn, inputs, targets, maxiter=100, verbose=False):
        """Optimize PyTorch model using L-BFGS."""
        for param in model.parameters():
            param.requires_grad_(True)
            
        optimizer = self._create_torch_optimizer(model.parameters())
        losses, grad_norms = [], []
        
        for step in range(maxiter):
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            # Calculate gradient norm
            total_norm = sum(p.grad.data.norm(2).item() ** 2 
                           for p in model.parameters() if p.grad is not None) ** 0.5
            
            losses.append(float(loss))
            grad_norms.append(total_norm)
            
            if verbose:
                logging.info(f"Step {step + 1}: Loss = {loss:.6f}, Grad = {total_norm:.6f}")
        
        self._loss_history.extend(losses)
        self._grad_norm_history.extend(grad_norms)
        
        return {
            'final_loss': losses[-1],
            'loss_history': losses,
            'grad_norm_history': grad_norms,
            'converged': grad_norms[-1] < self.tolerance_grad
        }
    
    def get_history(self):
        """Get optimization history."""
        return {'loss_history': self._loss_history.copy(), 
                'grad_norm_history': self._grad_norm_history.copy()}
    
    def reset_history(self):
        """Reset optimization history."""
        self._loss_history.clear()
        self._grad_norm_history.clear()