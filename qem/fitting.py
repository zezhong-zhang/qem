from curses import window
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import numpy as jnp
from jax import value_and_grad
from jax.example_libraries import optimizers
from scipy.ndimage import center_of_mass
from jaxopt import OptaxSolver
from skimage.feature import peak_local_max
from tqdm import tqdm
import copy

from .model import butterworth_window, gaussian_sum, voigt_sum, mask_grads, gaussian_local, add_gauss_to_total_sum, gaussian_global
from .utils import InteractivePlot, make_mask_circle_centre, find_duplicate_row_indices, remove_close_coordinates,get_random_indices_in_batches,find_element_indices

class ImageModelFitting:
    def __init__(self, image: np.array, pixel_size=1):
        """
        Initialize the Fitting class.

        Args:
            image (np.array): The input image as a numpy array.
            pixel_size (float, optional): The size of each pixel. Defaults to 1.
        """

        if len(image.shape) == 2:
            self.nx, self.ny = image.shape

        self.device = "cpu"
        self.image = image.astype(np.float32)
        self.local_shape = image.shape
        self.pixel_size = pixel_size
        self.atom_type = None
        self.coordinates = None
        self.fit_background = True
        self.same_width = True
        self.model = "gaussian"
        self.params = None
        self.fit_local = False
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

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
            window = butterworth_window(self.local_shape, 0.5, 10)
        else:
            window = butterworth_window(self.image.shape, 0.5, 10)
        return window

    def import_coordinates(self, coordinates, atom_type=None):
        """
        Import the coordinates of the atomic columns.

        Args:
            coordinates (np.array): The coordinates of the atomic columns.
            atom_type (np.array, optional): The type of each atomic column. Defaults to None.
        """

        if coordinates is not None:

            self.coordinates = coordinates
        if atom_type is not None:
            self.atom_type = atom_type
        else:
            self.atom_type = np.tile(0, self.num_coordinates)

    def find_peaks(
        self, atom_size=1, threshold_rel=0.2, exclude_border=False, image=None
    ):
        """
        Find the peaks in the image.

        Args:
            atom_size (int, optional): The size of the atomic columns. Defaults to 1.
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
            exclude_border=exclude_border,
        )
        peaks_locations = self.add_or_remove_peaks(
            peaks_locations, min_distance=min_distance, image=self.image
        )
        self.coordinates = peaks_locations
        return self.coordinates

    def remove_close_coordinates(self, threshold=10):
        self.coordinates = remove_close_coordinates(self.coordinates, threshold)
        return self.coordinates

    def add_or_remove_peaks(self, peaks_locations, min_distance=10, image=None):
        if image is None:
            image = self.image
        interactive_plot = InteractivePlot(
            peaks_locations=peaks_locations[:, [1, 0]],
            image=image,
            tolerance=min_distance,
        )
        interactive_plot.show()
        peaks_locations = [interactive_plot.pos_x, interactive_plot.pos_y]
        peaks_locations = np.array(peaks_locations).T.astype(float)
        return peaks_locations[:, [1, 0]]

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


    def refine_center_of_mass(self, plot=False):
        # do center of mass for each atom
        r, _, _ = self.guess_radius()
        windows_size = int(r) *2
        for i in range(self.num_coordinates):
            x, y = self.coordinates[i]
            x = int(x)
            y = int(y)
            # calculate the mask for distance < r
            region = self.image[
                x - windows_size : x + windows_size + 1,
                y - windows_size : y + windows_size + 1,
            ]
            mask = make_mask_circle_centre(region, r)
            region = (region-region.min())/(region.max()-region.min())
            region = region * mask
            local_x, local_y = center_of_mass(region)
            self.coordinates[i] = [
                x - windows_size + local_x,
                y - windows_size + local_y,
            ]
            if plot:
                plt.imshow(region, cmap="gray")
                plt.scatter(local_y, local_x, color='red',s = 2)
                plt.scatter(y%1+windows_size,x%1+windows_size, color='blue',s = 2)
                plt.show()
                plt.pause(1.0)
        return self.coordinates

    # def refine_local_max(self, plot=False, min_distance=10):
    #         windows_size = min_distance *5
    #         peak_total = np.array([], dtype=int).reshape(0, 2)
    #         for i in range(self.num_coordinates):
    #             x, y = self.coordinates[i]
    #             x = int(x)
    #             y = int(y)
    #             left = max(x - windows_size, 0)
    #             right = min(x + windows_size+1, self.nx)
    #             top = max(y - windows_size, 0)
    #             bottom = min(y + windows_size+1, self.ny)
    #             # calculate the mask for distance < r
    #             region = self.image[left : right, top : bottom]
    #             peaks_locations = peak_local_max(
    #                 region,
    #                 min_distance=int(0.8*min_distance),
    #                 threshold_rel=0.3,
    #                 exclude_border=True,
    #             )
    #             if peaks_locations.shape[0] > 0:
    #                 # select the peak position that is closest to the center
    #                 # distance = peaks_locations - np.array([windows_size, windows_size])
    #                 # distance = np.sqrt((distance**2).sum(axis=1))
    #                 # peak_select = peaks_locations[np.argmin(distance)]
    #                 # self.coordinates[i] = np.array([x - windows_size + peak_select[0], y - windows_size + peak_select[1]])
    #                 # append the peaks_locations to the peak_total
    #                 peak_total = np.append(peak_total, peaks_locations + np.array([x - windows_size, y - windows_size]), axis=0)
    #             if plot:
    #                 plt.imshow(region, cmap="gray")
    #                 plt.scatter(y%1+windows_size,x%1+windows_size, color='blue',s = 2)
    #                 if peaks_locations.shape[0] > 0:
    #                     plt.scatter(peaks_locations[:,1], peaks_locations[:,0], color='red',s = 2)
    #                 plt.show()
    #                 plt.pause(1.0)
    #         self.coordinates = np.unique(peak_total, axis=0)
    #         # self.coordinates = self.refine_duplicate_peaks()
    #         return self.coordinates




    # def refine_duplicate_peaks(self):
    #     duplicate_index = find_duplicate_row_indices(self.coordinates)
    #     # get the good peaks that are not duplicate
    #     good_peaks = np.delete(self.coordinates, duplicate_index, axis=0).astype(int)
    #     r, _, _ = self.guess_radius()
    #     windows_size = int(r) *5
    #     for x,y in self.coordinates[duplicate_index]:
    #         x = int(x)
    #         y = int(y)
    #         # calculate the mask for distance < r
    #         region = self.image[
    #             x - windows_size : x + windows_size + 1,
    #             y - windows_size : y + windows_size + 1,
    #         ]
    #         mask_neighbour = (good_peaks[:, 0] > x -  2* windows_size ) & (good_peaks[:, 0] < x + 2*windows_size) & (good_peaks[:, 1] > y - 2*windows_size) & (good_peaks[:, 1] < y + 2*windows_size)
    #         peaks_local = good_peaks[mask_neighbour]
    #         peaks_local = peaks_local - np.array([x - windows_size, y - windows_size])

    #         peaks_local_update = self.add_or_remove_peaks(
    #         peaks_local, min_distance=int(r), image=region
    #         )
    #         local_peaks_locations = peaks_local_update + np.array([x -2* windows_size, y -2* windows_size])
    #         # update the good_peaks
    #         good_peaks = np.append(good_peaks, local_peaks_locations, axis=0)
    #         good_peaks = np.unique(good_peaks, axis=0)                
    #         # peaks_locations = peak_local_max(
    #         #     region,
    #         #     min_distance=int(0.5*r),
    #         #     threshold_rel=0.2,
    #         #     exclude_border=False,
    #         # )
    #         # if peaks_locations.shape[0] > 0:
    #         #     # select the peak position that is closest to the center
    #         #     distance = peaks_locations - np.array([windows_size, windows_size])
    #         #     distance = np.sqrt((distance**2).sum(axis=1))
    #         #     peaks_locations_global = peaks_locations + np.array([x - windows_size, y - windows_size])
    #         #     # select the peak that is not in the self.coordinates and closest to the center
    #         #     mask = ~(peaks_locations_global[:, None] == good_peaks).all(-1).any(-1)

    #         #     if mask.any():
    #         #         peak_select = peaks_locations_global[mask][np.argmin(distance[mask])]
    #         #         peak_select_local = peak_select - np.array([x - windows_size, y - windows_size])
    #         #         good_peaks = np.append(good_peaks, [peak_select], axis=0)
    #         # if plot:
    #         #     plt.imshow(region, cmap="gray")
    #         #     # plot the peaks of neighbouring within the region
    #         #     mask_neighbour = (self.coordinates[:, 0] > x - 2* windows_size ) & (self.coordinates[:, 0] < x + 2*windows_size) & (self.coordinates[:, 1] > y - 2*windows_size) & (self.coordinates[:, 1] < y + 2*windows_size)
    #         #     plt.scatter(self.coordinates[mask_neighbour][:, 1] - (y - windows_size), self.coordinates[mask_neighbour][:, 0] - (x - windows_size), color='green',s = 2)
    #         #     if peak_select_local.shape[0] > 0:
    #         #         plt.scatter(peak_select_local[1], peak_select_local[0], color='red',s = 3)
    #         #     plt.show()
    #         #     plt.pause(1.0)    
    #     return good_peaks


    def plot(self, image="original"):
        plt.figure()
        # x = np.arange(self.nx) * self.pixel_size
        # y = np.arange(self.ny) * self.pixel_size
        if image == "original":
            plt.imshow(self.image, cmap="gray")
            plt.scatter(
                self.coordinates[:, 1], self.coordinates[:, 0], color="red", s=1
            )
        elif image == "prediction":
            plt.imshow(self.prediction, cmap="gray")
            plt.scatter(
                self.pos_y / self.pixel_size,
                self.pos_x / self.pixel_size,
                color="red",
                s=1,
            )
        plt.gca().set_aspect("equal", adjustable="box")

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

    def init_params(self):
        width = self.guess_radius()[0]
        # self.center_of_mass()
        coord = np.array(self.coordinates)
        index = (coord).astype(int)
        background = np.percentile(self.image, 20)
        height = self.image[index[:, 0], index[:, 1]].ravel() - background
        # get the lowest 20% of the intensity as the background
        width = np.tile(width, self.num_coordinates)
        if self.model == "gaussian":
            # Initialize the parameters
            params = {
                "pos_x": coord[:, 0],  # x position
                "pos_y": coord[:, 1],  # y position
                "height": height,  # height
                "sigma": width,  # width
            }
        elif self.model == "voigt":
            # Initialize the parameters
            params = {
                "pos_x": coord[:, 0],  # x position
                "pos_y": coord[:, 1],  # y position
                "height": height,  # height
                "sigma": width,  # width
                "gamma": width / np.log(2),  # width
                "ratio": 0.9,  # ratio
            }
        if self.fit_background:
            params["background"] = background

        return params

    def fit_region(self, params, fitting_region, update_region, maxiter=1000, tol=1e-4, step_size=0.01, plot=False,verbose=False):
        left,right,top,bottom = fitting_region
        center_left,center_right,center_top,center_bottom = update_region
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        # get the region of the image based on the patch_size and buffer_size
        # the buffer_size is on both sides of the patch

        image_region = self.image[left:right, top:bottom]
        self.local_shape = image_region.shape

        # get the region of the coordinates

        mask_region = (pos_x > left) & (pos_x < right) & (pos_y > top) & (pos_y < bottom)
        mask_center = (pos_x > center_left) & (pos_x < center_right) & (pos_y > center_top) & (pos_y < center_bottom)
        region_indices = np.where(mask_region)[0]
        center_indices = np.where(mask_center)[0]
        if len(center_indices) == 0:
            return params , None
        index_center_in_region = np.isin(region_indices, center_indices).squeeze()
        # index_center_in_region = find_element_indices(region_indices, center_indices)

        # get the buffer atoms as the difference between the region_atoms and the central_atoms
        local_X, local_Y = (
            self.X[left:right, top:bottom],
            self.Y[left:right, top:bottom],
        )
        if mask_center.sum() == 0:
            return params , None
        
        local_params = {
            key: value[mask_region]
            for key, value in params.items()
            if key not in ["background", "ratio"]
        }
        local_params["background"] = params["background"]
        if "ratio" in params:
            local_params["ratio"] = params["ratio"]

        global_prediction = self.predict(params, self.X, self.Y)
        local_prediction = self.predict(local_params, local_X, local_Y)
        local_residual = (
            global_prediction[left:right, top:bottom] - local_prediction
        )
        local_target = image_region - local_residual
        local_params = self.optimize(local_target, local_params, local_X, local_Y, maxiter, tol, step_size, verbose)
        # local_params = self.fit_gradient(image = local_target, params = local_params, X = local_X, Y = local_Y, step_size = step_size, maxiter = maxiter, tol = tol, keys_to_mask = ['background'])
        for key, value in local_params.items():
            if key not in ["background", "ratio"]:
                value_central = value[index_center_in_region]
                # check if the params[key] is jax array
                params[key][mask_center] = value_central
                # params[key][index_region_atoms] = value
            else:
                params[key] = value
        # local_params = self.fit_gradient(image = local_target, params = local_params, X = local_X, Y = local_Y, step_size = step_size, maxiter = maxiter, tol = tol, keys_to_mask = [])
        # else:
        #     local_params = self.fit_gradient(image = local_target, params = local_params, X = local_X, Y = local_Y, step_size = step_size, maxiter = maxiter, tol = tol, keys_to_mask = ['background', 'ratio','sigma'])

        if plot:
            plt.subplots(1, 3, figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image_region, cmap="gray")
            # plt.imshow(image_region, cmap="gray")
            plt.scatter(
                pos_y[mask_region] - top,
                pos_x[mask_region] - left,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_y[mask_center] - top,
                pos_x[mask_center] - left,
                color="r",
                s=1,
            )
            # get equal aspect ratio
            plt.gca().set_aspect("equal", adjustable="box")
            # plt.gca().invert_yaxis()
            plt.subplot(1, 3, 2)
            prediction = self.predict(local_params, local_X, local_Y)
            plt.imshow(prediction, cmap="gray")
            # mask = (local_params["pos_x"] > left) & (local_params["pos_x"] < right) & (local_params["pos_y"] > top) & (local_params["pos_y"] < bottom)
            plt.scatter(
                pos_y[mask_region] - top,
                pos_x[mask_region] - left,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_y[mask_center] - top,
                pos_x[mask_center] - left,
                color="r",
                s=1,
            )
            # plt.scatter(local_params["pos_y"][mask] - top, local_params["pos_x"][mask] - left, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.subplot(1, 3, 3)
            plt.imshow(local_target - prediction, cmap="gray")
            # plt.scatter(local_params["pos_y"][mask] - top, local_params["pos_x"][mask] - left, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")

            # plt.gca().invert_yaxis()
            plt.show()
        return params, local_params
    
    def fit_patch(
        self,
        params,
        step_size=0.01,
        maxiter=1000,
        tol=1e-4,
        patch_size=100,
        buffer_size=None,
        stride_size=None,
        plot=False,
        verbose=False,
        mode = "sequential",
        num_random_patches = 10,
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
            ii,jj = np.meshgrid(x_i, y_i, indexing='ij')
            ii = ii.ravel()
            jj = jj.ravel()
        elif mode == "random":
            ii = np.random.randint(half_patch, self.nx - half_patch, num_random_patches)
            jj = np.random.randint(half_patch, self.ny - half_patch, num_random_patches)

        for index in tqdm(range(len(ii))):
            i, j = ii[index], jj[index]
            left = max(i -half_patch - buffer_size, 0)
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
                print(f"left = {left}, right = {right}, top = {top}, bottom = {bottom}")
                print(f"center_left = {center_left}, center_right = {center_right}, center_top = {center_top}, center_bottom = {center_bottom}")
            params, local_params = self.fit_region(params, [left, right, top, bottom],[center_left,center_right,center_top,center_bottom], maxiter, tol, step_size, plot,verbose)
            if self.same_width:
                params['sigma'][:] =  params['sigma'].mean()
        # have a linear estimator of the background and height of the gaussian peaks
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)
        return params

    def optimize(self, image, params, X, Y, maxiter=1000, tol=1e-4, step_size=0.01, verbose=False):
        opt = optax.adam(step_size)
        solver = OptaxSolver(
                opt=opt, fun=self.loss, maxiter=maxiter, tol=tol, verbose=verbose
            )
        res = solver.run(params, image=image, X=X, Y=Y)
        params = res[0]
        return params

    def linear_estimator(self, params):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve, cg,lsqr
        # create the design matrix as array of gaussian peaks + background
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        sigma = params["sigma"]
        height = params["height"]
        rows = []
        cols = []
        data = []
        window_size = int(sigma.mean() * 5)      
        x = np.arange(-window_size, window_size+1, 1)
        y = np.arange(-window_size, window_size+1, 1)
        local_X, local_Y = np.meshgrid(x,y)
        gauss_local = gaussian_local(local_X, local_Y, pos_x, pos_y, height, sigma)

        for i in range(self.num_coordinates):
            global_X, global_Y = local_X + pos_x[i].astype(int), local_Y + pos_y[i].astype(int)
            mask = (global_X >= 0) & (global_X < self.nx) & (global_Y >= 0) & (global_Y < self.ny)
            flat_index = global_X[mask].flatten() * self.ny + global_Y[mask].flatten()
            rows.extend(flat_index)
            cols.extend(np.tile(i, flat_index.shape[0]))
            data.extend(gauss_local[:,:,i][mask].ravel())
        rows.extend(self.X.flatten() * self.ny + self.Y.flatten())
        cols.extend(np.tile(self.num_coordinates, self.nx*self.ny))
        data.extend(np.ones(self.nx*self.ny))
        design_matrix = coo_matrix((data, (rows, cols)), shape=(self.nx*self.ny, self.num_coordinates+1))
        # create the target as the image
        b = self.image.ravel()
        # solve the linear equation
        # solution = lsqr(design_matrix, b)[0]
        solution = spsolve(design_matrix.T @ design_matrix, design_matrix.T @ b)
        # solution = cg(design_matrix.T @ design_matrix, design_matrix.T @ b)[0]
        # update the background and height
        params["background"] = solution[-1] if solution[-1] > 0 else 0
        height_scale = solution[:-1]
        height_scale[height_scale < 0] = 1
        height_scale[height_scale > 2] = 2
        height_scale[height_scale < 0.5] = 0.5
        params["height"] = height_scale * params["height"] 
        return params

    
    def fit_global(self, maxiter=1000, tol=1e-4, step_size=0.01, verbose=False):
        params = self.init_params()
        params = self.optimize(self.image, params, self.X, self.Y, maxiter, tol, step_size, verbose)
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)

    def convergence(self, params, pre_params, tol=1e-2):
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
                print(f"Convergence rate for {key} = {max_update}")
                if max_update > 1:
                    print("Convergence not reached")
                    return False
            else:
                # Avoid division by zero and calculate relative update
                value_with_offset = value + 1e-10
                rate = np.abs(update / value_with_offset).max()
                print(f"Convergence rate for {key} = {rate}")
                if rate > tol:
                    print("Convergence not reached")
                    return False
                    
        print("Convergence reached")
        return True
                    
    def fit_random_batch(self, params, num_epoch =2, batch_size=500, maxiter=1000, tol=1e-4, step_size=0.01, verbose=False, plot = False):
        self.fit_local = False
        self.converged = False
        while self.converged is False and num_epoch > 0:
            num_epoch -= 1
            pre_params = copy.deepcopy(params)
            random_batches = get_random_indices_in_batches(self.num_coordinates, batch_size)
            image = self.image
            X = self.X
            Y = self.Y
            for index in tqdm(random_batches, desc="Fitting random batch"):
                mask = np.zeros(self.num_coordinates, dtype=bool)
                mask[index] = True
                select_params = {
                    key: value[mask]
                    for key, value in params.items()
                    if key not in ["background", "ratio"]
                }
                select_params["background"] = params["background"]
                if "ratio" in params:
                    select_params["ratio"] = params["ratio"]
                global_prediction = self.predict(params, self.X, self.Y)
                local_prediction = self.predict(select_params, self.X, self.Y)
                local_residual = (
                    global_prediction - local_prediction
                )
                local_target = image - local_residual
                select_params = self.optimize(local_target, select_params, X, Y, maxiter, tol, step_size, verbose)
                for key, value in select_params.items():
                    if key not in ["background", "ratio"]:
                        # check if the params[key] is jax array
                        params[key][mask] = value

                    else:
                        params[key] = value
                if self.same_width:
                    params['sigma'][:] =  params['sigma'].mean()
                if plot:
                    plt.subplots(1, 3, figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(local_target, cmap="gray")
                    # plt.scatter(
                    #     params["pos_y"], params["pos_x"], color="b", s=1
                    # )
                    plt.scatter(
                        params["pos_y"][index], params["pos_x"][index], color="r", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 2)
                    prediction = self.predict(select_params, X, Y)
                    plt.imshow(prediction, cmap="gray")
                    plt.scatter(
                        select_params["pos_y"], select_params["pos_x"], color="b", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 3)
                    plt.imshow(local_target - prediction, cmap="gray")
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.show()
            self.converged = self.convergence(params, pre_params, tol)
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)
        return params


    def fit_gradient(
        self,
        image,
        params,
        X,
        Y,
        step_size=0.001,
        maxiter=10000,
        tol=1e-4,
        keys_to_mask=None,
    ):
        opt_init, opt_update, get_params = optimizers.adam(
            step_size=step_size, b1=0.9, b2=0.999
        )
        opt_state = opt_init(params)

        def step(step_index,opt_state, params, image, X, Y, keys_to_mask=[]):
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
            loss_new, opt_state = step(i, opt_state, new_params, image, X, Y, keys_to_mask)  

            # Check if the loss has converged
            if np.abs(loss - loss_new) < tol * loss:
                break
            # Update the loss
            loss = loss_new
            loss_list.append(loss)

            # Print the loss every 10 iterations
            if i % 10 == 0:
                print(f"Iteration {i}: loss = {loss:.6f}")
        plt.plot(loss_list, "o-")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        # Update the model
        return new_params
    
    def l1_loss_smooth(self, predictions, targets, beta=1.0):

        loss = 0

        diff = predictions - targets
        mask = jnp.abs(diff) < beta
        loss += mask * (0.5 * diff**2 / beta)
        loss += (~mask) * (jnp.abs(diff) - 0.5 * beta)

        return loss.mean()

    def loss(self, params, image, X, Y):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = image - prediction
        diff = diff * self.window
        # diff = gaussian_filter_jax(diff, 2.0)

        # dammping the difference near the edge
        mse = jnp.sqrt(jnp.mean(diff ** 2))
        L1 = jnp.mean(jnp.abs(diff))
        # get the mse of binning differece
        # bin_size = 20
        # arr = diff
        # arr = arr[:arr.shape[0]//bin_size*bin_size,:arr.shape[1]//bin_size*bin_size]
        # arr = arr.reshape(arr.shape[0]//bin_size,bin_size,arr.shape[1]//bin_size,bin_size).mean(axis=1).mean(axis=2)
        # mse2 = np.std(arr)
        # L12 = jnp.mean(jnp.abs(arr))
        # L1 = self.l1_loss_smooth(predictions = prediction, targets= image, beta = 1.0)
        return mse + L1  # + mse2 + L12

    def residual(self, params, image, X, Y):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = prediction - image
        return diff
    
    def predict_local(self,params):
        if self.fit_background:
            background = params["background"]
        else:
            background = 0
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        height = params["height"]
        sigma = params["sigma"]
        if self.same_width:
            sigma = sigma.mean()
        windos_size = int(sigma.max()*3)
        x = np.arange(-windos_size, windos_size+1, 1)
        y = np.arange(-windos_size, windos_size+1, 1)
        X, Y = np.meshgrid(x,y)
        gauss_local = gaussian_local(X, Y, pos_x, pos_y, height, sigma)
        gauss_local = np.array(gauss_local)
        prediction = add_gauss_to_total_sum(np.zeros(self.image.shape), pos_x, pos_y, gauss_local, windos_size) + background
        return prediction
    
    def predict(self, params, X, Y):
        if self.fit_background:
            background = params["background"]
        else:
            background = 0
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        height = params["height"]
        sigma = params["sigma"]
        if self.same_width:
            sigma = sigma.mean()

        if self.model == "gaussian":
            # if self.num_coordinates<1000:
            prediction = gaussian_sum(X, Y, pos_x, pos_y, height, sigma, background)

        elif self.model == "voigt":
            gamma = params["gamma"]
            ratio = params["ratio"]
            if self.same_width:
                gamma = gamma.mean()

            prediction = voigt_sum(
                X,
                Y,
                pos_x,
                pos_y,
                height,
                sigma,
                gamma,
                ratio,
                background,
            )
        return prediction

    def update_params(self, params):
        self.pos_x = params["pos_x"] * self.pixel_size
        self.pos_y = params["pos_y"] * self.pixel_size
        self.height = params["height"]
        self.sigma = params["sigma"] * self.pixel_size
        if self.fit_background:
            self.background = params["background"]
        else:
            self.background = 0
        if self.model == "voigt":
            self.gamma = params["gamma"] * self.pixel_size
            self.ratio = params["ratio"]
        self.params = params

    @property
    def volume(self):
        if self.model == "gaussian":
            return self.height * self.sigma**2 * np.pi * 2
        elif self.model == "voigt":
            gaussian_contrib = self.height * self.sigma**2 * np.pi * 2 * self.ratio
            lorentzian_contrib = self.height * self.gamma * 2 * np.pi * (1 - self.ratio)
            return gaussian_contrib + lorentzian_contrib

    @property
    def num_coordinates(self):
        return self.coordinates.shape[0]
