import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

from qem.color import get_unique_colors

logging.basicConfig(level=logging.INFO)
import tkinter as tk
from tkinter import simpledialog

from matplotlib_scalebar.scalebar import ScaleBar

from qem.zoom import zoom_nd


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
    return atom_positions_selected, bool_array, path


def get_atom_type():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    atom_type = simpledialog.askstring(
        "Input", "Please enter the atom type:", parent=root
    )
    root.destroy()  # Close the main window
    return atom_type


class GetRegionSelection:
    def __init__(self, image, region_map, invert_selection=False):
        self.image = image
        self.region_map = region_map
        self.invert_selection = invert_selection
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(
            "Use the left mouse button to make the polygon.\n"
            "Click the first position to finish the polygon.\n"
            "Press ESC to reset the polygon, and hold SHIFT to\n"
            "move the polygon."
        )
        self.cax = self.ax.imshow(self.image)
        self.cax = self.ax.imshow(self.region_map, alpha=0.5)
        self.polygons = []  # Store all polygons
        self.poly = PolygonSelector(self.ax, self.onselect)
        self.fig.tight_layout()
        self.region_mask = np.zeros_like(self.image).astype(bool)

    def onselect(self, verts):
        self.path = Path(verts)
        self.verts = verts
        self.region_mask = self.get_region_mask()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_region_mask(self):
        points = np.indices(self.image.shape)
        points = points.reshape(2, -1).T
        region_mask = self.path.contains_points(points).reshape(self.image.shape).T
        if self.invert_selection:
            region_mask = ~region_mask
        return region_mask


class GetAtomSelection:
    def __init__(self, image, atom_positions, invert_selection=False, size=1):
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
        self.size = size
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
            self.atom_positions[:, 0],
            self.atom_positions[:, 1],
            "o",
            color="red",
            markersize=self.size,
        )[0]
        self.line_selected = None
        self.selection_mask = None
        self.region_mask = None
        handle_props = dict(color="blue")
        props = dict(color="blue")
        self.poly = PolygonSelector(
            self.ax, self.onselect, handle_props=handle_props, props=props  # type: ignore
        )
        self.fig.tight_layout()

    def onselect(self, verts):
        atom_positions_selected, selected, path = get_atom_selection_from_verts(
            self.atom_positions, verts, invert_selection=self.invert_selection
        )
        atom_positions_not_selected, not_selected, path = get_atom_selection_from_verts(
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

        if self.invert_selection:
            self.selection_mask = not_selected
            self.line_selected.set_data(  # type: ignore
                atom_positions_not_selected[:, 0], atom_positions_not_selected[:, 1]
            )
        else:
            self.selection_mask = selected
            self.line_selected.set_data(  # type: ignore
                atom_positions_selected[:, 0], atom_positions_selected[:, 1]
            )
        self.atom_positions_selected = atom_positions_selected

        self.path = path
        self.region_mask = self.get_mask_image(path, invert_selection=self.invert_selection)
        self.verts = verts
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_mask_image(self, path, invert_selection=False):
        points = np.indices(self.image.shape)
        points = points.reshape(2, -1).T
        region_mask = path.contains_points(points).reshape(self.image.shape).T
        if invert_selection:
            region_mask = ~region_mask
        return region_mask


class InteractivePlot:
    def __init__(
        self,
        image: np.ndarray,
        peaks_locations: np.ndarray,
        atom_types: np.ndarray = None,  # type: ignore
        tolerance: float = 10,
        dx: float = 1,
        units: str = "A",
        dimension: str = "si-length",
        zoom: float = 1,
        scale_y: float = 1,
    ):
        self.pos_x = peaks_locations[:, 0]
        self.pos_y = peaks_locations[:, 1]
        if atom_types is None:
            atom_types = np.zeros(len(peaks_locations))
        self.atom_types = atom_types
        self.image = image
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.tolerance = tolerance
        self.scatter_plot = None
        # Added attributes for vector selection
        self.origin = np.array([])
        self.vector_a = np.array([])
        self.vector_b = np.array([])
        self.current_atom_type = 0  # Default atom type is 0
        self.selection_stage = (
            0  # 0: Select origin, 1: Select vector a, 2: Select vector b
        )
        self.zoom = zoom
        self.scale_y = scale_y
        if zoom != 1:
            self.image = zoom_nd(image, upsample_factor=zoom)
            self.pos_x = self.pos_x * zoom - self.image.shape[1] // 2 * (zoom - 1)
            self.pos_y = self.pos_y * zoom - self.image.shape[0] // 2 * (zoom - 1)
            self.dx /= zoom

    def onclick_add_or_remove(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                index = np.argmin(distance)
                logging.info(
                    f"Removing peak at ({self.pos_x[index]}, {self.pos_y[index]})."
                )
                self.pos_x = np.delete(self.pos_x, index, axis=0)
                self.pos_y = np.delete(self.pos_y, index, axis=0)
                self.atom_types = np.delete(self.atom_types, index, axis=0)
            else:

                self.pos_x = np.append(self.pos_x, x)
                self.pos_y = np.append(self.pos_y, y)
                self.atom_types = np.append(self.atom_types, self.current_atom_type)
                logging.info(
                    f"Adding peak at ({x}, {y}) with atom type {self.current_atom_type}."
                )
            title = "Double click to add or remove peaks. Hit 'r' to reset the zoom."
            self.update_plot(title)

    def on_key_press(self, event):
        try:
            if event.key == "r":  # Check if the 'r' key is pressed
                self.scatter_plot = None  # Reset the scatter plot
                tiltle = (
                    "Double click to add or remove peaks. Hit 'r' to reset the zoom."
                )
                self.update_plot(tiltle)
            else:
                self.current_atom_type = int(
                    event.key
                )  # Try to set atom type if key is a number
                logging.info(f"Current atom type set to {self.current_atom_type}.")
        except ValueError:
            logging.info(
                f"Invalid atom type input: {event.key}. Atom type remains {self.current_atom_type}."
            )

    @property
    def scalebar(self):
        scalebar = ScaleBar(
            self.dx,
            units=self.units,
            location="lower right",
            dimension=self.dimension,
            box_alpha=0.5,
        )
        return scalebar

    def update_plot(self, title):
        if self.scatter_plot is not None:
            # Get current limits (view) of the plot
            xlim = self.scatter_plot.get_xlim()
            ylim = self.scatter_plot.get_ylim()
        plt.clf()
        plt.imshow(self.image, cmap="gray")
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.title(title)
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            plt.scatter(
                self.pos_x[mask],
                self.pos_y[mask],
                color=next(color_iterator),
                s=1,
                label=str(atom_type),
            )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if self.scatter_plot is not None:
            plt.gca().set_xlim(xlim)
            plt.gca().set_ylim(ylim)
        self.scatter_plot = plt.gca()
        plt.draw()

    def add_or_remove(self, tolerance: float = 10):
        self.tolerance = tolerance
        fig = plt.figure()
        title = "Double click to add or remove peaks. Press a number key to set the current atom type."
        self.update_plot(title)
        fig.canvas.mpl_connect("button_press_event", self.onclick_add_or_remove)
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        # plt.show()

        while plt.fignum_exists(fig.number):  # type: ignore
            plt.pause(0.1)

        logging.info("Updated peak locations.")

    def onclick_select(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                self.selected_point = (self.pos_x[i], self.pos_y[i])
                plt.scatter(
                    self.pos_x[i], self.pos_y[i], color="black", edgecolors="black", marker="x"
                )
                plt.draw()

    def select(self, tolerance: float = 10):
        self.tolerance = tolerance
        fig = plt.figure()
        plt.imshow(self.image, cmap="gray")
        title = "Double click to select a peak."
        self.update_plot(title)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select)
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.show()

        while plt.fignum_exists(fig.number):  # type: ignore
            plt.pause(0.1)

        logging.info(f"Selected peak location: {self.selected_point}.")
        return self.selected_point

    def onclick_select_vectors(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                point = np.array([self.pos_x[i], self.pos_y[i]])
                plt.scatter(
                    self.pos_x[i], self.pos_y[i], color="black", edgecolors="black", marker="x"
                )
                plt.draw()
            else:
                # If the point is not close to any peak, add it to the list
                point = np.array([x, y])
                self.pos_x = np.append(self.pos_x, x)
                self.pos_y = np.append(self.pos_y, y)
                self.atom_types = np.append(self.atom_types, 0)

            if self.selection_stage == 0:
                self.origin = point
                self.selection_stage += 1
                logging.info(f"Origin selected: {self.origin}", )
            elif self.selection_stage == 1:
                self.vector_a = point - self.origin
                self.selection_stage += 1
                logging.info(f"Vector a selected: {self.vector_a}")
                self.draw_arrow(self.origin, point, "a")
            elif self.selection_stage == 2:
                self.vector_b = point - self.origin
                logging.info(f"Vector b selected: {self.vector_b}")
                self.draw_arrow(self.origin, point, "b")
                self.selection_stage = (
                    0  # Reset the selection stage to allow new selections
                )
                return self.origin, self.vector_a, self.vector_b
            plt.draw()

    def draw_arrow(self, start, end, label):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(
            start[0],
            start[1],
            dx,
            dy,
            fc="black",
            ec="black",
            length_includes_head=True,
        )
        plt.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            label,
            color="black",
            fontsize=14,
        )

    def select_vectors(self, tolerance: float):
        self.tolerance = tolerance
        fig = plt.figure()
        plt.imshow(self.image, cmap="gray")
        title = "Double click to select origin, vector a, and vector b."
        self.update_plot(title)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select_vectors)
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.show()

        while plt.fignum_exists(fig.number):  # type: ignore
            plt.pause(0.1)
        selected = (
            self.origin.size > 0 and self.vector_a.size > 0 and self.vector_b.size > 0
        )

        if selected:
            if self.scale_y < 1:
                self.vector_a = self.vector_a * self.scale_y
                self.vector_b = self.vector_b * self.scale_y
            else:
                self.vector_a = self.vector_a / self.scale_y
                self.vector_b = self.vector_b / self.scale_y
            self.vector_a = self.vector_a/self.zoom
            self.vector_b = self.vector_b/self.zoom
            logging.info(
                f"In pixel: Origin: {self.origin}, Vector a: {self.vector_a}, Vector b: {self.vector_b}"
            )
            logging.info(
                f"In space: Origin: {self.origin*self.dx} {self.units}, Vector a: {self.vector_a*self.dx} {self.units}, Vector b: {self.vector_b*self.dx} {self.units}"
            )
            return self.origin, self.vector_a, self.vector_b
