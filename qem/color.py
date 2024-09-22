import itertools
import colorsys


def get_unique_colors():
    """
    An iterator to generate unique colors by cycling through the HSV color space.
    The saturation and value components are fixed to ensure bright and vivid colors.
    """
    # Choose a step size based on the number of elements you expect to plot to ensure distinct colors.
    # For a large number of elements, you may need to adjust the step size or modify the approach to generate more variations.
    step_size = 0.3
    for i in itertools.count():
        hue = (i * step_size) % 1.0
        # Convert HSV color to RGB since most plotting libraries use RGB.
        yield colorsys.hsv_to_rgb(hue, 1.0, 1.0)
