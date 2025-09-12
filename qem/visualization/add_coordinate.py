"""
Module for handling interactive coordinate addition and removal in matplotlib plots.
"""

import numpy as np


class AddCoordinate:
    """
    A class to handle the addition and removal of coordinates in a matplotlib plot.
    This class is typically used for interactive plotting where the user can click
    on the plot to add or remove coordinates.

    Attributes:
        line (matplotlib.lines.Line2D): The line object which is updated
            upon user interaction.
        init_coordinate (numpy.ndarray): Initial coordinates.
        extra_line (list): List of additional line objects to be updated.

    Methods:
        __call__(event): Method that handles mouse events; adds or removes coordinates
            based on user clicks.
    """

    def __init__(self, line, init_coordinate, extra_line):
        """
        Initializes the AddCoordinate object.

        Parameters:
            line (matplotlib.lines.Line2D): The line object in the
                matplotlib plot.
            init_coordinate (numpy.ndarray): Initial set of coordinates.
            extra_line (list): Additional line objects that need to be
                updated along with the main line.
        """
        self.line = line
        self.y, self.x = init_coordinate
        self.cid = line.figure.canvas.mpl_connect("button_press_event", self)
        self.extra_line = extra_line

    def __call__(self, event):
        """
        Method to handle mouse click events. Left-click adds a point, right-click
            removes the nearest point.

        Parameters:
            event: Mouse event object containing information about the click.
        """
        if event.button == 1:  # Left-click
            self.x = np.append(self.x, event.xdata)
            self.y = np.append(self.y, event.ydata)
        elif event.button == 3:  # Right-click
            dist = (self.x - event.xdata) ** 2 + (self.y - event.ydata) ** 2
            id_min = np.argmin(dist)
            self.x = np.delete(self.x, id_min)
            self.y = np.delete(self.y, id_min)

        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.draw()

        for el in self.extra_line:
            el.set_data(self.x, self.y)
            el.figure.canvas.draw()
