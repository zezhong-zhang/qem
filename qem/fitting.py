import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import numpy as jnp
from jax import value_and_grad
from jax.example_libraries import optimizers
# from scipy.ndimage import center_of_mass
from jaxopt import OptaxSolver
from skimage.feature import peak_local_max
from tqdm import tqdm

from .model import butterworth_window, gaussian_sum, voigt_sum
from .utils import InteractivePlot


class ImageModelFitting:
    def __init__(self, image: np.array, pixel_size=1):

        if len(image.shape) == 2:
            self.nx, self.ny = image.shape

        self.device = "cpu"
        self.image = image.astype(np.float32)
        self.local_shape = image.shape
        # self.image = (image - image.min()) / (image.max() - image.min())
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
        if self.fit_local:
            window = butterworth_window(self.local_shape, 0.4, 5)
        else:
            window = butterworth_window(self.image.shape, 0.5, 10)
        return window

    def import_coordinates(self, coordinates, atom_type=None):
        if coordinates is not None:

            self.coordinates = coordinates
        if atom_type is not None:
            self.atom_type = atom_type
        else:
            self.atom_type = np.tile(0, self.num_coordinates)

    def find_peaks(
        self, atom_size=1, threshold_rel=0.2, exclude_border=False, image=None
    ):
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

    def center_of_mass(self, num_peak=5):
        # do center of mass for each atom
        r, _, _ = self.guess_radius()
        windows_size = int(r) * 2
        for i in range(self.num_coordinates):
            x, y = self.coordinates[i]
            x = int(x)
            y = int(y)
            # calculate the mask for distance < r
            region = self.image[
                x - windows_size : x + windows_size + 1,
                y - windows_size : y + windows_size + 1,
            ]
            possible_shifts = np.zeros((num_peak, 2))
            region_rest = region.copy()
            for i in range(num_peak):
                possible_shifts[i, 0], possible_shifts[i, 1] = np.unravel_index(
                    np.argmax(region_rest), np.shape(region_rest)
                )
                left = max(int(possible_shifts[i, 0]) - int(r), 0)
                right = min(
                    int(possible_shifts[i, 0]) + int(r) + 1, windows_size * 2 + 1
                )
                top = max(int(possible_shifts[i, 1]) - int(r), 0)
                bottom = min(
                    int(possible_shifts[i, 1]) + int(r) + 1, windows_size * 2 + 1
                )
                region_rest[left:right, top:bottom] = 0
            # get the closest peak of possible_shifts to the center
            local_x, local_y = possible_shifts[
                np.argmin(np.linalg.norm(possible_shifts - windows_size, axis=1))
            ]
            # local_x, local_y = center_of_mass(region)
            # plt.imshow(region, cmap="gray")
            # plt.scatter(local_y, local_x, color='red',s = 2)
            # plt.scatter(y%1+windows_size,x%1+windows_size, color='blue',s = 2)
            # plt.show()
            # plt.pause(1.0)
            self.coordinates[i] = [
                x - windows_size + local_x,
                y - windows_size + local_y,
            ]
        return self.coordinates

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

    def init_params(self, gauss_ratio=0.9):
        width = self.guess_radius()[0]
        model = self.model
        # self.center_of_mass()
        coord = np.array(self.coordinates)
        index = (coord).astype(int)
        background = np.percentile(self.image, 20)
        height = self.image[index[:, 0], index[:, 1]].ravel() - background
        # get the lowest 20% of the intensity as the background
        width = np.tile(width, self.num_coordinates)
        if model == "gaussian":
            # Initialize the parameters
            params = {
                "pos_x": coord[:, 0],  # x position
                "pos_y": coord[:, 1],  # y position
                "height": height,  # height
                "width": width,  # width
            }
        elif model == "voigt":
            # Initialize the parameters
            params = {
                "pos_x": coord[:, 0],  # x position
                "pos_y": coord[:, 1],  # y position
                "height": height,  # height
                "width": width,  # width
                "gamma": width / np.log(2),  # width
                "ratio": gauss_ratio,  # ratio
            }
        if self.fit_background:
            params["background"] = background

        return params

    def segment_and_fit(
        self,
        model="gaussian",
        step_size=0.01,
        maxiter=1000,
        tol=1e-4,
        patch_size=100,
        buffer_size=None,
        plot=False,
    ):
        self.fit_local = True
        self.model = model
        params = self.init_params()
        atom_positions = self.coordinates
        if buffer_size is None:
            width, _, _ = (
                self.guess_radius()
            )  # Assuming width is a property of your class
            buffer_size = 3 * int(width)
        # stride_size = int(patch_size / 2)
        # create a sqaure patch with patch_size
        for i in tqdm(range(0, self.nx, patch_size)):
            for j in tqdm(range(0, self.ny, patch_size)):
                # get the region of the image based on the patch_size and buffer_size
                # the buffer_size is on both sides of the patch
                left = max(i - buffer_size, 0)
                right = min(i + patch_size + buffer_size, self.nx)
                top = max(j - buffer_size, 0)
                bottom = min(j + patch_size + buffer_size, self.ny)
                image_region = self.image[left:right, top:bottom]
                self.local_shape = image_region.shape

                center_left = max(i, 0)
                center_right = min(i + patch_size, self.nx)
                center_top = max(j, 0)
                center_bottom = min(j + patch_size, self.ny)
                # get the region of the coordinates
                region_atoms = atom_positions[
                    np.where(
                        (atom_positions[:, 0] > left)
                        & (atom_positions[:, 0] < right)
                        & (atom_positions[:, 1] > top)
                        & (atom_positions[:, 1] < bottom)
                    )
                ]
                central_atoms = atom_positions[
                    np.where(
                        (atom_positions[:, 0] > center_left)
                        & (atom_positions[:, 0] < center_right)
                        & (atom_positions[:, 1] > center_top)
                        & (atom_positions[:, 1] < center_bottom)
                    )
                ]
                # get the buffer atoms as the difference between the region_atoms and the central_atoms
                local_X, local_Y = (
                    self.X[left:right, top:bottom],
                    self.Y[left:right, top:bottom],
                )
                if len(central_atoms) == 0:
                    continue
                matches = (self.coordinates[:, None, :] == region_atoms).all(-1).any(1)
                index_region_atoms = np.where(matches)[0]
                matches = (self.coordinates[:, None, :] == central_atoms).all(-1).any(1)
                index_central_atoms = np.where(matches)[0]
                mask_central = np.isin(index_region_atoms, index_central_atoms)
                local_params = {
                    key: value[index_region_atoms]
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
                local_params = self.region_gradient(
                    local_target,
                    local_params,
                    local_X,
                    local_Y,
                    step_size=step_size,
                    maxiter=maxiter,
                    tol=tol,
                )

                if plot:
                    plt.subplots(1, 3, figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image_region, cmap="gray")
                    # plt.imshow(image_region, cmap="gray")
                    plt.scatter(
                        region_atoms[:, 1] - top,
                        region_atoms[:, 0] - left,
                        color="b",
                        s=1,
                    )
                    plt.scatter(
                        region_atoms[:, 1][mask_central] - top,
                        region_atoms[:, 0][mask_central] - left,
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
                        local_params["pos_y"] - top,
                        local_params["pos_x"] - left,
                        color="b",
                        s=1,
                    )
                    plt.scatter(
                        local_params["pos_y"][mask_central] - top,
                        local_params["pos_x"][mask_central] - left,
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

                # update the parameters only to the central atoms according to the mask_central
                for key, value in local_params.items():
                    if key not in ["background", "ratio"]:
                        i_x, i_y = local_params["pos_x"].astype(int), local_params["pos_y"].astype(int)
                        i_x = np.minimum(np.maximum(i_x - left, 0), right-left-1)
                        i_y = np.minimum(np.maximum(i_y - top, 0), bottom-top-1)
                        weight = self.window[i_x, i_y]
                        update  = params[key][index_region_atoms] - value
                        params[key][index_region_atoms] -= update * weight
                        # value_central = value[mask_central]
                        # check if the params[key] is jax array
                        # params[key][index_central_atoms] = value_central
                        # params[key][index_region_atoms] = value
                    else:
                        weight = len(index_region_atoms)/self.num_coordinates
                        params[key] = value*weight + params[key]*(1-weight)
        # params = self.global_optimize(params)
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)

    def fit(self, maxiter=1000, tol=1e-4, step_size=0.01, verbose=False):
        # if self.params is None:
        params = self.init_params()
        # elif len(self.params['pos_x']) != self.num_coordinates:
        #     params = self.init_params()
        # else:
        #     params = self.params
        # use the jaxopt to optimize the parameters using the BFGS
        opt = optax.adam(step_size)
        solver = OptaxSolver(
            opt=opt, fun=self.loss, maxiter=maxiter, tol=tol, verbose=verbose
        )
        # solver = jaxopt.LevenbergMarquardt(self.residual, maxiter=1000, tol=tol)
        # solver = jaxopt.GradientDescent(fun=self.loss, maxiter=500)
        res = solver.run(params, image=self.image, X=self.X, Y=self.Y)
        params = res[0]
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)

    def region_gradient(
        self,
        image_region,
        local_params,
        local_X,
        local_Y,
        step_size=0.001,
        maxiter=1000,
        tol=1e-4,
        plot=False,
        verbose=False,
    ):
        # Initialize the optimizer
        opt_init, opt_update, get_params = optimizers.adam(
            step_size=step_size, b1=0.9, b2=0.999
        )
        opt_state = opt_init(local_params)
        # Initialize the loss
        loss = np.inf
        loss_list = []
        # loss_list = []
        # Loop over the number of iterations
        for i in range(maxiter):
            # Update the parameters
            new_params = get_params(opt_state)
            loss_new, opt_state = self.step(
                i, opt_state, opt_update, new_params, image_region, local_X, local_Y
            )
            # Check if the loss has converged
            if np.abs(loss - loss_new) < tol * loss:
                break
            # Update the loss
            loss = loss_new
            loss_list.append(loss)
            # Print the loss every 10 iterations

            if i % 10 == 0 and verbose:
                print(f"Iteration {i}: loss = {loss:.6f}")
        if plot:
            plt.plot(loss_list, "o-")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
        # Update the model
        new_params = get_params(opt_state)
        return new_params

    def step(self, step_index, opt_state, opt_update, new_params, image, X, Y):
        loss, grads = value_and_grad(self.loss)(new_params, image, X, Y)
        opt_state = opt_update(step_index, grads, opt_state)
        return loss, opt_state

    def fit_gradient(
        self,
        step_size=0.001,
        maxiter=1000,
        tol=1e-4,
        model="gaussian",
        optimizer="adam",
    ):
        # Detect the peaks
        self.model = model
        params = self.init_params()
        # Initialize the optimizer
        # if optimizer == "LBFGS":
        #     solver = jaxopt.LBFGS(self.loss, maxiter=maxiter)
        #     res = solver.run(params)
        #     params = res[0]

        if optimizer == "adam":
            opt_init, opt_update, get_params = optimizers.adam(
                step_size=step_size, b1=0.9, b2=0.999
            )
            opt_state = opt_init(params)

            # Initialize the loss
            loss = np.inf
            loss_list = []
            # Loop over the number of iterations
            for i in range(maxiter):
                new_params = get_params(opt_state)
                # Update the parameters
                loss_new, opt_state = self.step(
                    i, opt_state, opt_update, new_params, self.image, self.X, self.Y
                )

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
            params = get_params(opt_state)
        self.update_params(params)
        self.prediction = self.predict(params, self.X, self.Y)

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
        # x = jnp.linspace(-2, 2, 10)
        # window = jax.scipy.stats.norm.pdf(x) * jax.scipy.stats.norm.pdf(x[:, None])
        # diff = jax.scipy.signal.convolve2d(diff, window, mode="same")

        diff = diff * self.window

        # dammping the difference near the edge
        mse = np.std(diff)
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

    def predict(self, params, X, Y):
        if self.fit_background:
            background = params["background"]
        else:
            background = 0
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        height = params["height"]
        sigma = params["width"]
        if self.same_width:
            sigma = sigma.mean()

        if self.model == "gaussian":
            # if self.num_coordinates<1000:
            prediction = gaussian_sum(X, Y, pos_x, pos_y, height, sigma, background)
            # else:
            #     windos_size = int(sigma.max()*3)
            #     x = np.arange(-windos_size, windos_size+1, 1)
            #     y = np.arange(-windos_size, windos_size+1, 1)
            #     X, Y = np.meshgrid(x,y)
            #     gauss_local = gaussian_local(X, Y, pos_x, pos_y, height, sigma)
            #     # if the type of the pos_x is jax trace array
            #     # if isinstance(pos_x, jax.core.Tracer):
            #     #     gauss_local = np.array(gauss_local.val)
            #     #     pos_x = np.array(pos_x.val)
            #     #     pos_y = np.array(pos_y.val)
            #     #     background = background.val
            #     # elif isinstance(pos_x, np.ndarray):
            #     #     gauss_local = np.array(gauss_local)
            #     #     pos_x = np.array(pos_x)
            #     #     pos_y = np.array(pos_y)
            #     prediction = add_gauss_to_total_sum_jax(jnp.zeros(self.image.shape), pos_x, pos_y, gauss_local, windos_size) + background
            # # the the residual size devide by the batch size
            # batch_size = 100
            # res_size = len(pos_x) % batch_size
            # # get the the number of batches
            # num_batches = len(pos_x) // batch_size
            # # get the size of all the batches
            # total_batch = num_batches * batch_size
            # prediction = gaussian_sum_batched(
            # X, Y, pos_x[:total_batch], pos_y[:total_batch], height[:total_batch], sigma, background
            # )
            # if res_size > 0:
            #     prediction += gaussian_sum(
            #         X,
            #         Y,
            #         pos_x[total_batch:],
            #         pos_y[total_batch:],
            #         height[total_batch:],
            #         sigma,
            #         0,
            #     )

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
        self.sigma = params["width"] * self.pixel_size
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
