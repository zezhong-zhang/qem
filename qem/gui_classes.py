import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import logging

logging.basicConfig(level=logging.INFO)

def get_atom_selection_from_verts(atom_positions, verts, invert_selection=False):
    """Get a subset of atom positions within region spanned to verticies.

    Parameters
    ----------
    atom_positions : list or NumPy array
        In the form [[x0, y0]. [x1, y1], ...]
    verts : list of tuples
        List of positions, spanning an enclosed region.
        [(x0, y0), (x1, y1), ...]. Need to have at least 3 positions.
    invert_selection : bool, optional
        Get the atom positions outside the region, instead of the
        ones inside it. Default False.

    Returns
    -------
    atom_positions_selected : NumPy array

    Examples
    --------
    >>> from numpy.random import randint
    >>> from atomap.tools import _get_atom_selection_from_verts
    >>> atom_positions = randint(0, 200, size=(200, 2))
    >>> verts = [(200, 400), (400, 600), (100, 100)]
    >>> atom_positions_selected = _get_atom_selection_from_verts(
    ...        atom_positions=atom_positions, verts=verts)

    Get atom positions inside the region

    >>> atom_positions_selected = _get_atom_selection_from_verts(
    ...        atom_positions=atom_positions, verts=verts,
    ...        invert_selection=True)

    """
    if len(verts) < 3:
        raise ValueError(
            "verts needs to have at least 3 positions, not {0}".format(len(verts))
        )
    atom_positions = np.array(atom_positions)
    path = Path(verts)
    bool_array = path.contains_points(atom_positions)
    if invert_selection:
        bool_array = np.invert(bool_array)
    atom_positions_selected = atom_positions[bool_array]
    return atom_positions_selected

class GetAtomSelection:
    def __init__(self, image, atom_positions, invert_selection=False):
        """Get a subset of atom positions using interactive tool.

        Access the selected atom positions in the
        atom_positions_selected attribute.

        Parameters
        ----------
        image : 2D HyperSpy signal or 2D NumPy array
        atom_positions : list of lists, NumPy array
            In the form [[x0, y0]. [x1, y1], ...]
        invert_selection : bool, optional
            Get the atom positions outside the region, instead of the
            ones inside it. Default False

        Attributes
        ----------
        atom_positions_selected : NumPy array

        """
        self.image = image
        self.atom_positions = np.array(atom_positions)
        self.invert_selection = invert_selection
        self.atom_positions_selected = np.ndarray(shape=(0, 2))
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(
            "Use the left mouse button to make the polygon.\n"
            "Click the first position to finish the polygon.\n"
            "Press ESC to reset the polygon, and hold SHIFT to\n"
            "move the polygon."
        )
        self.cax = self.ax.imshow(self.image)
        self.line_non_selected = self.ax.plot(
            self.atom_positions[:, 0], self.atom_positions[:, 1], "o", color="red"
        )[0]
        self.line_selected = None
        handle_props = dict(color="blue")
        props = dict(color="blue")
        self.poly = PolygonSelector(
            self.ax, self.onselect, handle_props=handle_props, props=props
        )
        self.fig.tight_layout()

    def onselect(self, verts):
        atom_positions_selected = get_atom_selection_from_verts(
            self.atom_positions, verts, invert_selection=self.invert_selection
        )
        atom_positions_not_selected = get_atom_selection_from_verts(
            self.atom_positions, verts, invert_selection=not self.invert_selection
        )
        if len(atom_positions_selected) != 0:
            if self.line_selected is None:
                self.line_selected = self.ax.plot(
                    atom_positions_selected[:, 0],
                    atom_positions_selected[:, 1],
                    "o",
                    color="green",
                )[0]
            else:
                self.line_selected.set_data(
                    atom_positions_not_selected[:, 0], atom_positions_not_selected[:, 1]
                )
            self.line_selected.set_data(
                atom_positions_selected[:, 0], atom_positions_selected[:, 1]
            )

        self.atom_positions_selected = atom_positions_selected
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



class InteractivePlot:
    def __init__(self, peaks_locations, image, tolerance):
        self.pos_x = peaks_locations[:, 0]
        self.pos_y = peaks_locations[:, 1]
        self.image = image
        self.tolerance = tolerance
        self.scatter_plot = None
        # Added attributes for vector selection
        self.origin = None
        self.vector_a = None
        self.vector_b = None
        self.selection_stage = 0  # 0: Select origin, 1: Select vector a, 2: Select vector b

    def onclick_add_or_remove(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                logging.info(f"Removing peak at ({self.pos_x[i]}, {self.pos_y[i]}).")
                self.pos_x = np.delete(self.pos_x, i, axis=0)
                self.pos_y = np.delete(self.pos_y, i, axis=0)
            else:
                self.pos_x = np.append(self.pos_x, x)
                self.pos_y = np.append(self.pos_y, y)
                logging.info(f"Adding peak at ({x}, {y}).")
            
            self.update_plot()

    def update_plot(self):
        plt.clf()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)
        plt.draw()

    def add_or_remove(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_add_or_remove)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        print("Updated peak locations.")

    def onclick_select(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                self.selected_point = (self.pos_x[i], self.pos_y[i])
                plt.scatter(self.pos_x[i], self.pos_y[i], color="red", s=5)
                plt.draw()

    def select(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="blue", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        print(f"Selected peak location: {self.selected_point}.")
        return self.selected_point

    def onclick_select_vectors(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                point = np.array([self.pos_x[i], self.pos_y[i]])
                plt.scatter(self.pos_x[i], self.pos_y[i], color="red", s=10)
                plt.draw()

                if self.selection_stage == 0:
                    self.origin = point
                    self.selection_stage += 1
                    print("Origin selected:", self.origin)
                elif self.selection_stage == 1:
                    self.vector_a = point - self.origin
                    self.selection_stage += 1
                    print("Vector a selected:", self.vector_a)
                    self.draw_arrow(self.origin, point, 'a')
                elif self.selection_stage == 2:
                    self.vector_b = point - self.origin
                    print("Vector b selected:", self.vector_b)
                    self.draw_arrow(self.origin, point, 'b')
                    self.selection_stage = 0  # Reset the selection stage to allow new selections
                    return self.origin, self.vector_a, self.vector_b
            plt.draw()

    def draw_arrow(self, start, end, label):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(start[0], start[1], dx, dy, head_width=10, head_length=10, fc='black', ec='black')
        plt.text((start[0] + end[0])/2, (start[1] + end[1])/2, label, color='black', fontsize=14)

    def select_vectors(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="blue", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select_vectors)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        if self.origin.any() and self.vector_a.any() and self.vector_b.any():
            print(f"Origin: {self.origin}, Vector a: {self.vector_a}, Vector b: {self.vector_b}")
            return self.origin, self.vector_a, self.vector_b
        else:
            print("Selection incomplete.")
            return None, None, None
