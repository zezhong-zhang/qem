"""
Plotting utilities for QEM visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, x_labels, y_labels, colormap="gray", colorbar=True):
    """
    Plot an image. with x and y labels.

    Parameters
    ----------
    image : numpy.ndarray
        The image to plot.
    x_labels : list
        The labels for the x-axis.
    y_labels : list
        The labels for the y-axis.
    colormap : str, optional
        The colormap to use.
    colorbar : bool, optional
        Whether to show a colorbar.

    Returns
    -------
    matplotlib.pyplot.figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(
        image,
        cmap=colormap,
        extent=[x_labels[0], x_labels[-1], y_labels[0], y_labels[-1]],
        origin="lower",
    )
    if colorbar:
        fig.colorbar(ax.images[0])
    return fig