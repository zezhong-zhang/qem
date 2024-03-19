import copy
import logging
import warnings
from curses import window

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import jit
from functools import partial

from jax import numpy as jnp
from jax import value_and_grad
from jax.example_libraries import optimizers
from jaxopt import OptaxSolver
from skimage.feature.peak import peak_local_max
from tqdm import tqdm
import copy
import logging
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import warnings
from qem.model import (
    butterworth_window,
    gaussian_parallel,
    voigt_parallel,
    mask_grads,
    gaussian_local,
    add_gaussian_at_positions,
)
from qem.utils import (
    InteractivePlot,
    make_mask_circle_centre,
    remove_close_coordinates,
    get_random_indices_in_batches,
)
from scipy.ndimage import center_of_mass
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
class ImageModelFitting:
    def __init__(self, image: np.ndarray, pixel_size: float = 1.0):
        """
        Initialize the Fitting class.

        Args:
            image (np.array): The input image as a numpy array.
            pixel_size (float, optional): The size of each pixel. Defaults to 1.
        """

        if len(image.shape) == 2:
            self.ny, self.nx = image.shape

        self.device = "cpu"
        self.image = image.astype(np.float32)
        self.model = np.zeros(image.shape)
        self.local_shape = image.shape
        self.pixel_size = pixel_size
        self._atom_types = np.array([])
        self.atoms_selected = np.array([])
        self.coordinates = np.array([])
        self.fit_background = True
        self.same_width = True
        self.fitting_model = "gaussian"
        self.params = dict()
        self.fit_local = False
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")


### Properties
    @property
    def window(self):
        """
        Returns the window used for fitting.

        If `fit_local` is True, a Butterworth window is created with the shape of `local_shape`.
        If `fit_local` is False, a Butterworth window is created with the shape of `image.shape`.

        Returns:
            numpy.ndarray: The Butterworth window used for fitting.
        """
        if self.fit_local:
            return butterworth_window(self.local_shape, 0.5, 10)
        else:
            return butterworth_window(self.image.shape, 0.5, 10)

    
    @property
    def atom_types(self):
        if self._atom_types.shape != self.num_coordinates:
            self._atom_types = np.zeros(self.num_coordinates, dtype=int)
        return self._atom_types 
    
    @property
    def volume(self):
        params = self.params
        if self.fitting_model == "gaussian":
            return params["height"] * params["sigma"]**2 * np.pi * 2 *self.pixel_size**2
        elif self.fitting_model == "voigt":
            gaussian_contrib = params["height"] * params["sigma"]**2 * np.pi * 2 * params["ratio"] *self.pixel_size**2
            lorentzian_contrib = params["height"] * params["gamma"] * 2 * np.pi * (1 - params["ratio"]) *self.pixel_size**2
            return gaussian_contrib + lorentzian_contrib

    @property
    def voronoi_volume(self):
        if self._voronoi_volume is None:
            self.voronoi_integration()
        return self._voronoi_volume

    @property
    def num_coordinates(self):
        return self.coordinates.shape[0]

### voronoi integration
    def voronoi_integration(self,plot=False):
        """
        Compute the Voronoi integration of the atomic columns.

        Returns:
            np.array: The Voronoi integration of the atomic columns.
        """
        from hyperspy.signals import Signal2D
        from qem.voronoi import integrate
        s = Signal2D(self.image - self.params["background"])
        pos_x = self.params["pos_x"]
        pos_y = self.params["pos_y"]
        max_radius = self.params["sigma"].max() * 5
        integrated_intensity, intensity_record, point_record = integrate(s, pos_x, pos_y, max_radius=max_radius)
        integrated_intensity = integrated_intensity * self.pixel_size**2
        intensity_record = intensity_record * self.pixel_size**2
        self._voronoi_volume = integrated_intensity
        self._voronoi_map = intensity_record
        self._voronoi_cell = point_record
        if plot:
            intensity_record.plot(cmap='viridis')
        return integrated_intensity, intensity_record, point_record

### init peaks and parameters
    def import_coordinates(self, coordinates: np.ndarray):
        """
        Import the coordinates of the atomic columns.

        Args:
            coordinates (np.array): The coordinates of the atomic columns.
        """
        self.coordinates = coordinates

    def find_peaks(
        self, atom_size:float=1, threshold_rel:float=0.2, threshold_abs = None, exclude_border:bool=False, image=None
    ):
        """
        Find the peaks in the image.

        Args:
            atom_size (float, optional): The size of the atomic columns. Defaults to 1.
            threshold_rel (float, optional): The relative threshold. Defaults to 0.2.
            exclude_border (bool, optional): Whether to exclude the border. Defaults to False.
            image (np.array, optional): The input image. Defaults to None.

        Returns:
            np.array: The coordinates of the peaks.
        """
        if image is None:
            image = self.image
        min_distance = int(atom_size / self.pixel_size)
        peaks_locations = peak_local_max(
            image,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            exclude_border=exclude_border,
        )
        self.coordinates = peaks_locations[:,[1,0]].astype(float)
        self.add_or_remove_peaks(min_distance=min_distance, image=self.image
        )
        self._atom_types = np.zeros(self.num_coordinates, dtype=int)
        return self.coordinates

    def remove_close_coordinates(self, threshold:int=10):
        self.coordinates = remove_close_coordinates(self.coordinates, threshold)
        return self.coordinates

    def add_or_remove_peaks(self, min_distance:int=10, image=None):
        if image is None:
            image = self.image
        peaks_locations = self.coordinates
        interactive_plot = InteractivePlot(
            peaks_locations=peaks_locations,
            image=image,
            tolerance=min_distance,
        )
        interactive_plot.show()
        peaks_locations = [interactive_plot.pos_x, interactive_plot.pos_y]
        peaks_locations = np.array(peaks_locations).T.astype(float)
        return peaks_locations

    def remove_peaks_outside_image(self):
        coordinates = self.coordinates
        mask = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] <= self.nx)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] <= self.ny)
        )
        self.coordinates = coordinates[mask]
        return self.coordinates

    def plot(self):
        plt.figure(figsize=(10, 5))
        # x = np.arange(self.nx) * self.pixel_size
        # y = np.arange(self.ny) * self.pixel_size
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap="gray")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.subplot(1, 2, 2)
        plt.hist(self.image.ravel(), bins=256)

    def guess_radius(self):
        """
        Estimate the density of atomic columns in an image.

        Parameters:
        id (int): Identifier for a specific image or set of coordinates.

        Returns:
        tuple: density, influence_map, background_region
        """
        num_coordinates = self.coordinates.shape[0]
        if num_coordinates == 0:
            raise ValueError("No coordinates found for the given id.")

        rate, rate_max, n_filled, n = 1, 1, 0, 0
        nx, ny = self.image.shape

        while rate > 0.5 * rate_max:
            influence_map = np.zeros((nx, ny))
            for i in range(num_coordinates):
                i_l = np.maximum(self.coordinates[i, 0] - n, 0).astype(np.int64)
                i_r = np.minimum(self.coordinates[i, 0] + n, self.nx).astype(np.int64)
                i_u = np.maximum(self.coordinates[i, 1] - n, 0).astype(np.int64)
                i_d = np.minimum(self.coordinates[i, 1] + n, self.ny).astype(np.int64)
                influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1
            if n == 0:
                rate = (np.sum(influence_map) - n_filled) / num_coordinates
            else:
                rate = (np.sum(influence_map) - n_filled) / (8 * n) / num_coordinates
            n_filled = np.sum(influence_map)
            rate_max = max(rate_max, rate)
            n += 1

        # Scaled factors
        n1 = int(np.round((n - 1) * 10))
        n2 = int(np.round((n - 1) * 1))

        influence_map = np.zeros((nx, ny))
        direct_influence_map = np.zeros((nx, ny))

        for i in range(num_coordinates):
            # Calculate the indices for the larger area (influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n1, 0).astype(np.int64)
            i_r = np.minimum(self.coordinates[i, 0] + n1, nx).astype(np.int64)
            i_u = np.maximum(self.coordinates[i, 1] - n1, 0).astype(np.int64)
            i_d = np.minimum(self.coordinates[i, 1] + n1, ny).astype(np.int64)
            influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

            # Calculate the indices for the smaller area (direct_influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n2, 0).astype(np.int64)
            i_r = np.minimum(self.coordinates[i, 0] + n2, nx).astype(np.int64)
            i_u = np.maximum(self.coordinates[i, 1] - n2, 0).astype(np.int64)
            i_d = np.minimum(self.coordinates[i, 1] + n2, ny).astype(np.int64)
            direct_influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

        radius = (np.sum(direct_influence_map) / num_coordinates) ** (1 / 2) / np.pi

        background_region = influence_map - direct_influence_map
        return radius, direct_influence_map, background_region

    def init_params(self, atom_size:float=0.7, guess_radius:bool=True):            
        if guess_radius:
            width = self.guess_radius()[0]
        else:
            width = atom_size / self.pixel_size
        # self.center_of_mass()
        pos_x = copy.deepcopy(self.coordinates[:, 0])
        pos_y = copy.deepcopy(self.coordinates[:, 1])
        background = np.percentile(self.image, 20)
        height = self.image[pos_y.astype(int), pos_x.astype(int)].ravel() - background
        # get the lowest 20% of the intensity as the background
        width = np.tile(width, self.num_coordinates)
        ratio = np.tile(0.9, self.num_coordinates)
        if self.fitting_model == "gaussian":
            # Initialize the parameters
            params = {
                "pos_x": pos_x,  # x position
                "pos_y": pos_y,  # y position
                "height": height,  # height
                "sigma": width,  # width
            }
        elif self.fitting_model == "voigt":
            # Initialize the parameters
            params = {
                "pos_x": pos_x,  # x position
                "pos_y": pos_y,  # y position
                "height": height,  # height
                "sigma": width,  # width
                "gamma": width / np.log(2),  # width
                "ratio": ratio,  # ratio
            }
        if self.fit_background:
            params["background"] = background.astype(float)

        self.params = params
        return params

### loss function and model prediction

    def loss(self, params:dict, image:np.ndarray, X:np.ndarray, Y:np.ndarray):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = image - prediction
        diff = diff * self.window
        # dammping the difference near the edge
        mse = jnp.sqrt(jnp.mean(diff**2))
        L1 = jnp.mean(jnp.abs(diff))
        return mse + L1

    def residual(self, params:dict, image:np.ndarray, X:np.ndarray, Y:np.ndarray):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = prediction - image
        return diff

    def predict_local(self, params:dict):
        if self.fit_background:
            background = params["background"]
        else:
            background = 0
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        height = params["height"]
        sigma = params["sigma"]
        windos_size = int(sigma.max() * 5)
        x = np.arange(-windos_size, windos_size + 1, 1)
        y = np.arange(-windos_size, windos_size + 1, 1)
        X, Y = np.meshgrid(x, y, indexing="xy")
        gauss_local = gaussian_local(X, Y, pos_x, pos_y, height, sigma)
        gauss_local = np.array(gauss_local)
        prediction = (
            add_gaussian_at_positions(
                np.zeros(self.image.shape), pos_x, pos_y, gauss_local, windos_size
            )
            + background
        )
        prediction = (
            add_gaussian_at_positions(
                np.zeros(self.image.shape), pos_x, pos_y, gauss_local, windos_size
            )
            + background
        )
        return prediction

    def predict(self, params:dict, X:np.ndarray, Y:np.ndarray):
        if self.fit_background:
            background = params["background"]
        else:
            background = 0
        if self.fitting_model == "gaussian":
            # if self.num_coordinates<1000:
            prediction = gaussian_parallel(
                X,
                Y,
                params["pos_x"],
                params["pos_y"],
                params["height"],
                params["sigma"],
                background,
            )

        elif self.fitting_model == "voigt":
            prediction = voigt_parallel(
                X,
                Y,
                params["pos_x"],
                params["pos_y"],
                params["height"],
                params["sigma"],
                params["gamma"],
                params["ratio"],
                background,
            )
        return prediction

### fitting

    def linear_estimator(self, params:dict):
        # create the design matrix as array of gaussian peaks + background
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        sigma = params["sigma"]
        height = params["height"]
        rows = []
        cols = []
        data = []
        window_size = int(sigma.mean() * 5)
        x = np.arange(-window_size, window_size + 1, 1)
        y = np.arange(-window_size, window_size + 1, 1)
        local_X, local_Y = np.meshgrid(x, y, indexing="xy")
        gauss_local = gaussian_local(local_X, local_Y, pos_x, pos_y, height, sigma)

        for i in range(self.num_coordinates):
            global_X, global_Y = local_X + pos_x[i].astype(int), local_Y + pos_y[
                i
            ].astype(int)
            mask = (
                (global_X >= 0)
                & (global_X < self.nx)
                & (global_Y >= 0)
                & (global_Y < self.ny)
            )
            flat_index = global_Y[mask].flatten() * self.nx + global_X[mask].flatten()
            rows.extend(flat_index)
            cols.extend(np.tile(i, flat_index.shape[0]))
            data.extend(gauss_local[:, :, i][mask].ravel())
        rows.extend(self.Y.flatten() * self.nx + self.X.flatten())
        cols.extend(np.tile(self.num_coordinates, self.nx * self.ny))
        data.extend(np.ones(self.nx * self.ny))
        design_matrix = coo_matrix(
            (data, (rows, cols)), shape=(self.nx * self.ny, self.num_coordinates + 1)
        )
        # create the target as the image
        b = self.image.ravel()
        # solve the linear equation
        # solution = lsqr(design_matrix, b)[0]
        try:
            # Attempt to solve the linear system
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                solution = spsolve(design_matrix.T @ design_matrix, design_matrix.T @ b)
                # Check if any of the caught warnings are related to a singular matrix
                if w and any(
                    "singular matrix" in str(warning.message) for warning in w
                ):
                    logging.warning(
                        "Warning: Singular matrix encountered. Please refine the peak positions better before linear estimation. The parameters are not updated."
                    )
                    return params
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                logging.warning("Error: Singular matrix encountered.")
            else:
                raise

        # solution = cg(design_matrix.T @ design_matrix, design_matrix.T @ b)[0]
        # update the background and height
        height_scale = solution[:-1]
        if np.NaN in height_scale:
            logging.warning(
                "The height has NaN, the linear estimator is not valid, parameters are not updated"
            )
            return params
        else:
            params["background"] = solution[-1] if solution[-1] > 0 else 0.0
            # if (height_scale>2).any():
            #     logging.warning(
            #         "The height has values larger than 2, the linear estimator is probably not accurate. I will limit it to 2 but be careful with the results."
            #     )
            #     height_scale[height_scale>2] =2
            # if (height_scale < 0.5).any():
            #     logging.warning(
            #         "The height has values smaller than 0.5, the linear estimator is probably not accurate. I will limit it to 0.5 but be careful with the results."
            #     )
            #     mask = (height_scale < 0.5) & (height_scale > 0)
            #     height_scale[mask] = 0.5

            if (height_scale * params["height"] < 0).any():
                logging.warning(
                    "The height has negative values, the linear estimator is not valid. I will make it positive but be careful with the results."
                )
                input_negative_mask = params["height"] < 0
                scale_neagtive_mask = height_scale < 0
                height_scale[scale_neagtive_mask] = 1
                params["height"][input_negative_mask] = -params["height"][input_negative_mask]
            params["height"] = height_scale* params["height"]
        self.params = params
        return params

    def optimize(
        self, image:np.ndarray, params:dict, X:np.ndarray, Y:np.ndarray, maxiter:int=1000, tol:float=1e-4, step_size:float=0.01, verbose:bool=False
    ):
        opt = optax.adam(learning_rate=step_size)
        solver = OptaxSolver(
            opt=opt, fun=self.loss, maxiter=maxiter, tol=tol, verbose=verbose
        )
        res = solver.run(params, image=image, X=X, Y=Y)
        params = res[0]
        return params

    def gradient_descent(
        self,
        image:np.ndarray,
        params:dict,
        X:np.ndarray,
        Y:np.ndarray,
        keys_to_mask:list[str],
        step_size:float=0.001,
        maxiter:int=10000,
        tol:float=1e-4,
    ):
        opt_init, opt_update, get_params = optimizers.adam(
            step_size=step_size, b1=0.9, b2=0.999
        )
        opt_state = opt_init(params)

        def step(step_index, opt_state, params, image, X, Y, keys_to_mask=[]):
            loss, grads = value_and_grad(self.loss)(params, image, X, Y)
            masked_grads = mask_grads(grads, keys_to_mask)
            opt_state = opt_update(step_index, masked_grads, opt_state)
            return loss, opt_state

        # Initialize the loss
        loss = np.inf
        loss_list = []
        # Loop over the number of iterations
        for i in range(maxiter):
            # Update the parameters
            new_params = get_params(opt_state)
            loss_new, opt_state = step(
                i, opt_state, new_params, image, X, Y, keys_to_mask
            )

            # Check if the loss has converged
            if np.abs(loss - loss_new) < tol * loss:
                break
            # Update the loss
            loss = loss_new
            loss_list.append(loss)

            # Print the loss every 10 iterations
            if i % 10 == 0:
                logging.info(f"Iteration {i}: loss = {loss:.6f}")
        plt.plot(loss_list, "o-")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        # Update the model
        self.params = new_params
        return new_params

    def fit_global(self, params: dict, maxiter:int=1000, tol:float=1e-3, step_size:float=0.01, verbose:bool=False):
        self.fit_local = False
        params = self.optimize(
            self.image, params, self.X, self.Y, maxiter, tol, step_size, verbose
        )
        params = self.same_width_on_atom_type(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

    def fit_random_batch(
        self,
        params: dict,
        num_epoch:int=5,
        batch_size:int=500,
        maxiter:int=50,
        tol:float=1e-3,
        step_size:float=1e-2,
        verbose:bool=False,
        plot:bool=False,
    ):
        self.fit_local = False
        self.converged = False
        while self.converged is False and num_epoch > 0:
            params = self.linear_estimator(params)
            pre_params = copy.deepcopy(params)
            num_epoch -= 1
            random_batches = get_random_indices_in_batches(
                self.num_coordinates, batch_size
            )
            image = self.image
            X = self.X
            Y = self.Y

            for index in tqdm(random_batches, desc="Fitting random batch"):
                mask = np.zeros(self.num_coordinates, dtype=bool)
                mask[index] = True
                params = self.same_width_on_atom_type(params)
                select_params = self.select_params(params, mask)
                global_prediction = self.predict(params, self.X, self.Y)
                local_prediction = self.predict(select_params, self.X, self.Y)
                local_residual = global_prediction - local_prediction
                local_target = image - local_residual
                select_params = self.optimize(
                    local_target, select_params, X, Y, maxiter, tol, step_size, verbose
                )
                select_params = self.project_params(select_params)
                params = self.update_from_local_params(params, select_params, mask)
                if plot:
                    plt.subplots(1, 3, figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image, cmap="gray")
                    # plt.scatter(
                    #     params["pos_x"], params["pos_y"], color="b", s=1
                    # )
                    plt.scatter(
                        params["pos_x"][index], params["pos_y"][index], color="r", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 2)
                    plt.imshow(global_prediction, cmap="gray")
                    plt.scatter(
                        select_params["pos_x"], select_params["pos_y"], color="b", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 3)
                    plt.imshow(image - global_prediction, cmap="gray")
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.show()
            params = self.same_width_on_atom_type(params)
            self.converged = self.convergence(params, pre_params, tol)
        params = self.linear_estimator(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

    def fit_region(
        self,
        params: dict,
        fitting_region: list,
        update_region: list,
        maxiter:int=1000,
        tol:float=1e-4,
        step_size:float=0.01,
        plot:bool=False,
        verbose:bool=False,
    ):
        self.fit_local = True
        left, right, top, bottom = fitting_region
        center_left, center_right, center_top, center_bottom = update_region
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        # get the region of the image based on the patch_size and buffer_size
        # the buffer_size is on both sides of the patch

        image_region = self.image[top:bottom, left:right]
        self.local_shape = image_region.shape

        # get the region of the coordinates

        mask_region = (
            (pos_x > left) & (pos_x < right) & (pos_y > top) & (pos_y < bottom)
        )
        mask_center = (
            (pos_x > center_left)
            & (pos_x < center_right)
            & (pos_y > center_top)
            & (pos_y < center_bottom)
        )
        region_indices = np.where(mask_region)[0]
        center_indices = np.where(mask_center)[0]
        if len(center_indices) == 0:
            return params, None
        index_center_in_region = np.isin(region_indices, center_indices).squeeze()
        # index_center_in_region = find_element_indices(region_indices, center_indices)

        # get the buffer atoms as the difference between the region_atoms and the central_atoms
        local_X, local_Y = (
            self.X[top:bottom, left:right],
            self.Y[top:bottom, left:right],
        )
        if mask_center.sum() == 0:
            return params, None
        local_params = self.select_params(params, mask_region)
        self.atoms_selected = mask_region
        global_prediction = self.predict(params, self.X, self.Y)
        local_prediction = self.predict(local_params, local_X, local_Y)
        local_residual = global_prediction[top:bottom, left:right] - local_prediction
        local_target = image_region - local_residual
        local_params = self.optimize(
            local_target,
            local_params,
            local_X,
            local_Y,
            maxiter,
            tol,
            step_size,
            verbose,
        )
        params = self.update_from_local_params(params, local_params, mask_center, index_center_in_region)
        # params = self.update_from_local_params(params, local_params, mask_region)
        if plot:
            plt.subplots(1, 3, figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image_region, cmap="gray")
            # plt.imshow(image_region, cmap="gray")
            plt.scatter(
                pos_x[mask_region] - left,
                pos_y[mask_region] - top,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_x[mask_center] - left,
                pos_y[mask_center] - top,
                color="r",
                s=1,
            )
            # get equal aspect ratio
            plt.gca().set_aspect("equal", adjustable="box")
            # plt.gca().invert_yaxis()
            plt.subplot(1, 3, 2)

            plt.imshow(local_prediction, cmap="gray")
            # mask = (local_params["pos_x"] > left) & (local_params["pos_x"] < right) & (local_params["pos_y"] > top) & (local_params["pos_y"] < bottom)
            plt.scatter(
                pos_x[mask_region] - left,
                pos_y[mask_region] - top,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_x[mask_center] - left,
                pos_y[mask_center] - top,
                color="r",
                s=1,
            )
            # plt.scatter(local_params["pos_x"][mask] - left, local_params["pos_y"][mask] - top, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.subplot(1, 3, 3)
            plt.imshow(local_target - local_prediction, cmap="gray")
            # plt.scatter(local_params["pos_x"][mask] - left, local_params["pos_y"][mask] - top, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")

            # plt.gca().invert_yaxis()
            plt.show()
        return params, local_params

    def fit_patch(
        self,
        params: dict,
        step_size:float=0.01,
        maxiter:int=1000,
        tol:float=1e-4,
        patch_size:int=100,
        buffer_size:int=0,
        stride_size:int=100,
        plot:bool=False,
        verbose:bool=False,
        mode:str="sequential",
        num_random_patches:int=10,
    ):
        self.fit_local = True
        if buffer_size is None:
            width, _, _ = (
                self.guess_radius()
            )  # Assuming width is a property of your class
            buffer_size = 5 * int(width)
        if stride_size is None:
            stride_size = int(patch_size / 2)
        # create a sqaure patch with patch_size
        half_patch = int(patch_size / 2)
        if mode == "sequential":
            x_i = np.arange(half_patch, self.nx, stride_size)
            y_i = np.arange(half_patch, self.ny, stride_size)
            ii, jj = np.meshgrid(x_i, y_i, indexing="xy")
            ii = ii.ravel()
            jj = jj.ravel()
        elif mode == "random":
            ii = np.random.randint(half_patch, max(self.nx - half_patch, patch_size), num_random_patches)
            jj = np.random.randint(half_patch, max(self.ny - half_patch, patch_size), num_random_patches)
        params = self.linear_estimator(params)
        for index in tqdm(range(len(ii))):
            i, j = ii[index], jj[index]
            left = max(i - half_patch - buffer_size, 0)
            right = min(i + half_patch + buffer_size, self.nx)
            top = max(j - half_patch - buffer_size, 0)
            bottom = min(j + half_patch + buffer_size, self.ny)

            if left < buffer_size:
                left = 0
            if right > self.nx - buffer_size:
                right = self.nx
            if top < buffer_size:
                top = 0
            if bottom > self.ny - buffer_size:
                bottom = self.ny

            center_left = left + buffer_size if left > 0 else 0
            center_right = right - buffer_size if right < self.nx else self.nx
            center_top = top + buffer_size if top > 0 else 0
            center_bottom = bottom - buffer_size if bottom < self.ny else self.ny
            if verbose:
                logging.info(f"left = {left}, right = {right}, top = {top}, bottom = {bottom}")
                logging.info(
                    f"center_left = {center_left}, center_right = {center_right}, center_top = {center_top}, center_bottom = {center_bottom}"
                )
            params, local_params = self.fit_region(
                params,
                [left, right, top, bottom],
                [center_left, center_right, center_top, center_bottom],
                maxiter,
                tol,
                step_size,
                plot,
                verbose,
            )

        # have a linear estimator of the background and height of the gaussian peaks
        self.same_width_on_atom_type(params)
        params = self.linear_estimator(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

### parameters updates and convergence
    def convergence(self, params:dict, pre_params:dict, tol:float=1e-2):
        """
        Checks if the parameters have converged within a specified tolerance.

        This function iterates over each parameter in `params` and its corresponding
        value in `pre_params` to determine if the change (update) is within a specified
        tolerance level, `tol`. For position parameters ('pos_x', 'pos_y'), it checks if
        the absolute update exceeds 1. For other parameters ('height', 'sigma', 'gamma',
        'ratio', 'background'), it checks if the relative update exceeds `tol`.

        Parameters:
            params (dict): Current values of the parameters.
            pre_params (dict): Previous values of the parameters.
            tol (float, optional): Tolerance level for convergence. Default is 1e-2.

        Returns:
            bool: True if all parameters have converged within the tolerance, False otherwise.
        """
        # Loop through current parameters and their previous values
        for key, value in params.items():
            if key not in pre_params:
                continue  # Skip keys that are not in pre_params

            # Calculate the update difference
            update = np.abs(value - pre_params[key])

            # Check convergence based on parameter type
            if key in ["pos_x", "pos_y"]:
                max_update = update.max()
                logging.info(f"Convergence rate for {key} = {max_update}")
                if max_update > 1:
                    logging.info("Convergence not reached")
                    return False
            else:
                # Avoid division by zero and calculate relative update
                value_with_offset = value + 1e-10
                rate = np.abs(update / value_with_offset).mean()
                logging.info(f"Convergence rate for {key} = {rate}")
                if rate > tol:
                    logging.info("Convergence not reached")
                    return False

        logging.info("Convergence reached")
        return True

    def select_params(self, params:dict, mask:np.ndarray):
        select_params = {
            key: value[mask]
            for key, value in params.items()
            if key not in ["background"]
            }
        select_params["background"] = params["background"]
        return select_params

    def update_from_local_params(self, params:dict, local_params:dict, mask:np.ndarray, mask_local=None):
        for key, value in local_params.items():
            value = np.array(value)
            if key not in ["background"]:
                if mask_local is None:
                    params[key][mask] = value
                else:
                    params[key][mask] = value[mask_local]
            else:
                weight = mask.sum() / self.num_coordinates
                update = value - params[key]
                params[key] += update*weight
                # params[key] = value
        return params
    
    def same_width_on_atom_type(self, params:dict):
        if self.same_width:
            unique_types = np.unique(self.atom_types)
            for atom_type in unique_types:
                mask = self.atom_types == atom_type
                mask = mask.squeeze()
                for key, value in params.items():
                    is_jax_traced = isinstance(params[key], jax.numpy.ndarray)
                    if key in ["sigma", "gamma", "ratio"]:
                        if is_jax_traced:
                            # Calculate the mean only for the masked values using JAX
                            mean_value = jnp.mean(value[mask])
                            # Use JAX's indexing to update the values
                            params[key] = value.at[mask].set(mean_value)
                        else:
                            # For non-JAX arrays, we can proceed with NumPy operations
                            mean_value = np.mean(value[mask])
                            params[key][mask] = mean_value
        return params

    def project_params(self, params:dict):
        for key, value in params.items():
            if key == "pos_x":
                params[key] = jnp.clip(value, 0, self.nx - 1)
            elif key == "pos_y":
                params[key] = jnp.clip(value, 0, self.ny - 1)
            elif key == "height":
                params[key] = jnp.clip(value, 0, np.sum(self.image))
            elif key == "sigma":
                params[key] = jnp.clip(value, 1, min(self.nx, self.ny) / 2)
            elif key == "gamma":
                params[key] = jnp.clip(value, 1, min(self.nx, self.ny) / 2)
            elif key == "ratio":
                params[key] = jnp.clip(value, 0, 1)
            elif key == "background":
                params[key] = jnp.clip(value, 0, np.max(self.image))
        return params
