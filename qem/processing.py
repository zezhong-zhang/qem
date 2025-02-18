import numpy as np

def butterworth_window(shape, cutoff_radius_ftr, order):
    """
    Generate a 2D Butterworth window.

    Parameters:
    - shape: tuple of ints, the shape of the window (height, width).
    - cutoff_radius_ftr: float, the cutoff frequency as a fraction of the radius (0, 0.5].
    - order: int, the order of the Butterworth filter.

    Returns:
    - window: 2D numpy array, the Butterworth window.
    """
    assert len(shape) == 2, "Shape must be a tuple of length 2 (height, width)"
    assert (
        0 < cutoff_radius_ftr <= 0.5
    ), "Cutoff frequency must be in the range (0, 0.5]"

    def butterworth_1d(length, cutoff_radius_ftr, order):
        n = np.arange(-np.floor(length / 2), length - np.floor(length / 2))
        return 1 / (1 + (n / (cutoff_radius_ftr * length)) ** (2 * order))

    window_y = butterworth_1d(shape[0], cutoff_radius_ftr, order)
    window_x = butterworth_1d(shape[1], cutoff_radius_ftr, order)

    window = np.outer(window_y, window_x)

    return window
