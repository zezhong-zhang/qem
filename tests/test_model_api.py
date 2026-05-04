#!/usr/bin/env python3
"""Simple script to test model API."""

try:
    from qem.utils.backend import setup_test_backend
    setup_test_backend()
    
    import torch
    from qem.fit.model import GaussianModel
    from qem.utils import safe_convert_to_numpy
    
    print("Testing model API...")
    
    # Create a simple grid
    x_grid = torch.arange(10, dtype=torch.float32)
    y_grid = torch.arange(10, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    
    # Create test parameters
    params = {
        "pos_x": torch.as_tensor([5.0], dtype=torch.float32),
        "pos_y": torch.as_tensor([5.0], dtype=torch.float32),
        "height": torch.as_tensor([1.0], dtype=torch.float32),
        "width": torch.as_tensor([2.0], dtype=torch.float32),
        "background": torch.as_tensor(0.1, dtype=torch.float32)
    }
    
    # Test model
    model = GaussianModel(dx=1.0)
    model.set_params(params)
    model.build()
    
    # Test sum method
    result = model.sum(x_grid, y_grid, local=False)
    result_np = safe_convert_to_numpy(result)
    
    print(f"Model output shape: {result_np.shape}")
    print(f"Model output max: {result_np.max()}")
    print(f"Model output min: {result_np.min()}")
    
    # Test volume calculation
    volumes = model.volume(params)
    volumes_np = safe_convert_to_numpy(volumes)
    print(f"Volume: {volumes_np}")
    
    print("Model API test successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()