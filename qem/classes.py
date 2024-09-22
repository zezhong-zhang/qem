import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import curve_fit, minimize, fmin, Bounds
from scipy.optimize import curve_fit
from qem.utils import safe_ln, fft2d, ifft2d
from qem import utils
from tqdm import tqdm
from multiprocessing import Pool
import timeit
import math
from numba import jit
from qem.archieve.model_numba import (
    gaussian,
    gaussian_sum_same_sigma,
    gaussian_sum_different_sigma,
    calc_fit_outcome,
    loss,
)
from tqdm import tqdm

# import partial


class Image:
    """
    This class is used to store the experimental data.
    """

    data: np.ndarray  # 2D array of experimental data.
    dx: float  # pixel size in x direction
    dy: float  # pixel size in y direction
    signal: str  # String that represents the signal.
    shape: tuple  # Shape of the image

    def __init__(
        self,
        data,
        signal={"HAADF", "ADF", "ABF", "BF", "iDPC", "iCOM"},
        dx=1.0,
        dy=None,
    ):
        """
        Parameters
        ----------
        data : array_like
            2D array of experimental data.
        signal : set
            A set of strings that represent the signal.
        """
        self.data = data
        self.signal = signal
        self.shape = data.shape
        self.dx = dx
        self.dy = dy if dy is not None else dx

    def plot(self, colormap="gray", colorbar=True):
        """
        Plot the image.

        Parameters
        ----------
        colormap (optional): str
            Colormap to use.
        colorbar (optional): bool
            Whether to show a colorbar.
        """
        fig = utils.plot_image(
            self.data,
            [0, self.dx * self.shape[1]],
            [0, self.dy * self.shape[0]],
            colormap=colormap,
            colorbar=colorbar,
        )
        fig.axes[0].set_title(self.signal)
        plt.show(block=True)


class AtomicColumns:
    """
    Class to hold atomic column data which can have multiple signals and corrsponding
    scattering cross sections but unique positions.
    """

    signals: list
    scs: list  # scs value (float) for each signal
    pos: np.ndarray  # atom column positions shared amoung all signals

    def __assert_signal_scs__(self, signal, scs):
        if isinstance(signal, list):
            n_signals = len(signal)
        else:
            n_signals = 1
            signal = [signal]

        if isinstance(scs, list):
            n_scs = len(scs)
        else:
            n_scs = 1
            scs = [scs]
        assert n_signals == n_scs, "Signals and scs must have same length."

        for i in range(n_signals):
            assert (
                signal[i] not in self.signals
            ), "Duplicate signal identifiers not allowed."

        return signal, scs

    def __init__(self, signals, scs, pos):
        """
        Parameters
        ----------
        signals : list
            List of strings that represent the signals.
        scs : dict of np.ndarray
            List of scattering cross sections for each signal.
            The order must correspond to the order in the signals list.
        pos : array_like
            Array of atom column positions.
        """
        self.scs = {}
        self.signals = []
        signals, scs = self.__assert_signal_scs__(signals, scs)
        self.signals = signals
        for i in range(len(signals)):
            self.scs[signals[i]] = scs[i]
        assert len(self.scs[signals[i]]) == len(
            pos
        ), "AtomicColumns.__init__: scs and pos must be the same length"
        self.pos = pos

    def add_signal(self, signals, scs):
        """
        Add a signal to the data.

        Parameters
        ----------
        signals : list
            List of strings that represent the signals.
        scs : list of np.ndarray
            List of scattering cross sections for each signal.
            The order must correspond to the order in the signals list.
        """
        signals, scs = self.__assert_signal_scs__(signals, scs)
        self.signals.append(signals)
        for i in range(len(signals)):
            self.scs[signals[i]] = scs[i]


class Experiment:
    """
    This class is used to store the experimental data.
    """

    images: list  # list of `Image` objects
    dx: float
    dy: float
    xy_grid: np.ndarray
    signals: list
    columns: list  # list of `AtomicColumns` objects

    @staticmethod
    def __image2list__(image):
        if not isinstance(image, list):
            image = [image]
        return image

    def __init__(self, images):
        """
        Parameters
        ----------
        images : list of `Image`
            list of 2D images of experimental data.
        """

        self.images = []
        self.signals = []
        images = self.__image2list__(images)
        for image in images:
            assert isinstance(
                image, Image
            ), "Experiment.__init__: images must be a list of Image objects."
            assert (
                image.shape == images[0].shape
            ), "Experiment.__init__: all images must have the same shape."
            assert np.allclose(
                [image.dx, image.dy], [images[0].dx, images[0].dy]
            ), "Experiment.__init__: all images must have the same pixel size."
            self.images.append(image)
            self.signals.append(image.signal)
        self.dx = images[0].dx
        self.dy = images[0].dy
        self.xy_grid = np.meshgrid(
            np.arange(images[0].shape[1]), np.arange(images[0].shape[0])
        )

    def plot(self, colormap="gray", colorbar=True):
        """
        Plot the image.

        Parameters
        ----------
        colormap (optional): str
            Colormap to use.
        colorbar (optional): bool
            Whether to show a colorbar.
        """
        fig, ax = plt.subplots(1, len(self.images))
        for i in range(len(self.images)):
            im = ax[i].imshow(
                self.images[i].data,
                cmap=colormap,
                extent=[
                    0,
                    self.dx * self.images[i].shape[1],
                    0,
                    self.dy * self.images[i].shape[0],
                ],
                origin="lower",
            )
            ax[i].set_title(self.signals[i])
            if colorbar:
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show(block=True)


class TimeSeries:
    """
    This class is used to store time series data.
    """

    experiments: list  # list of `Experiment` objects.

    def __init__(self, experiments):
        """
        Parameters
        ----------
        data : array_like
            list of experimental data of type `Experiment`.
        """
        self.experiments = []
        for experiment in experiments:
            assert isinstance(
                experiment, Experiment
            ), "TimeSeries.__init__: data must be a list of Experiment objects."
            self.experiments.append(experiment)


################################################################################################################################
class ImageProcess:
    def __init__(self, image: np.ndarray, coor=None):
        if image.ndim == 2:
            self.nx, self.ny = image.shape
            self.nz = 1
            image = np.reshape(image, [self.nz, self.nx, self.ny])
        else:
            self.nz, self.nx, self.ny = image.shape
        self.image = image
        self.image_cropped = np.zeros(image.shape)
        self.image_processed = np.zeros(image.shape)
        self.image_fitted = np.zeros(image.shape)
        self.th_dist = 0
        self.th_inten = 0
        self.coordinate = []
        if coor is not None:
            self.coordinate.append(coor)
        self.use_coor = []
        self.gaussian_prmt = []
        for i in range(self.nz):
            self.gaussian_prmt.append([])

    @staticmethod
    def apply_threshold(image, image_ref, threshold):
        nz = np.size(image, 0)
        if type(threshold) != list:
            threshold = [threshold]
        img = np.zeros(image.shape)
        for i in range(nz):
            m = np.amax(image_ref[i])
            img[i, :, :] = np.where(image_ref[i] < threshold[i] * m, 0, image[i])
        return img

    @staticmethod
    def remove_freq(image, low, high):
        nx, ny = image.shape[1:]
        nx = nx
        ny = ny
        x = np.linspace(-nx / 2, nx / 2, nx) / nx
        y = np.linspace(-ny / 2, ny / 2, ny) / ny
        yv, xv = np.meshgrid(y, x)
        mask = np.where(
            (np.sqrt(xv**2 + yv**2) >= low) * (np.sqrt(xv**2 + yv**2) < high),
            1,
            0,
        )
        return np.real(ifft2d(fft2d(image) * mask))

    @staticmethod
    def __local_max__(array):
        if np.argmax(array) == 4:
            return 1
        else:
            return 0

    def local_max(self, image):
        nz, nx, ny = image.shape
        map = np.zeros(image.shape)
        for z in range(nz):
            for x in range(nx - 2):
                for y in range(ny - 2):
                    map[z, x + 1, y + 1] = self.__local_max__(
                        image[z, x : x + 3, y : y + 3].ravel()
                    )
        return map

    def __guess_density__(self, id):
        n_coor = np.size(self.coordinate[id], 1)
        rate = 1
        rate_max = 1
        n_filled = 0
        n = 0
        while rate > 0.5 * rate_max:
            map = np.zeros(self.image[id].shape)
            for i in range(n_coor):
                i_l = np.maximum(self.coordinate[id][0, i] - n, 0).astype(np.int64)
                i_r = np.minimum(self.coordinate[id][0, i] + n, self.nx).astype(
                    np.int64
                )
                i_u = np.maximum(self.coordinate[id][1, i] - n, 0).astype(np.int64)
                i_d = np.minimum(self.coordinate[id][1, i] + n, self.ny).astype(
                    np.int64
                )
                map[i_l : i_r + 1, i_u : i_d + 1] = 1
            if n == 0:
                rate = (np.sum(map) - n_filled) / n_coor
            else:
                rate = (np.sum(map) - n_filled) / (8 * n) / n_coor
            n_filled = np.sum(map)
            rate_max = np.maximum(rate_max, rate)
            n += 1

        map = np.zeros(self.image[id].shape)
        map2 = np.zeros(self.image[id].shape)
        n1 = int(np.round((n - 1) * 10))
        n2 = int(np.round((n - 1) * 3))
        for i in range(n_coor):
            i_l = np.maximum(self.coordinate[id][0, i] - n1, 0).astype(np.int64)
            i_r = np.minimum(self.coordinate[id][0, i] + n1, self.nx).astype(np.int64)
            i_u = np.maximum(self.coordinate[id][1, i] - n1, 0).astype(np.int64)
            i_d = np.minimum(self.coordinate[id][1, i] + n1, self.ny).astype(np.int64)
            map[i_l : i_r + 1, i_u : i_d + 1] = 1
            i_l = np.maximum(self.coordinate[id][0, i] - n2, 0).astype(np.int64)
            i_r = np.minimum(self.coordinate[id][0, i] + n2, self.nx).astype(np.int64)
            i_u = np.maximum(self.coordinate[id][1, i] - n2, 0).astype(np.int64)
            i_d = np.minimum(self.coordinate[id][1, i] + n2, self.ny).astype(np.int64)
            map2[i_l : i_r + 1, i_u : i_d + 1] = 1
        density = np.sum(map2) / n_coor
        return density, map, map - map2

    def __interactive_add_remove__(self, i):
        fig, axs = plt.subplots(1, 3)
        fig.dpi = 150
        axs[0].imshow(self.image[i])
        f1 = axs[0].plot(
            self.coordinate[i][1], self.coordinate[i][0], ".w", markersize=1
        )
        axs[1].imshow(self.image_processed[i])
        f2 = axs[1].plot(
            self.coordinate[i][1], self.coordinate[i][0], ".w", markersize=1
        )
        f3 = axs[2].plot(
            self.coordinate[i][1], self.coordinate[i][0], ".k", markersize=1
        )
        axs[2].set_aspect("equal")
        axs[2].set(xlim=(0, self.ny), ylim=(0, self.nx))
        axs[2].invert_yaxis()
        addcoor = AddCoordinate(f3[0], self.coordinate[i], [f1[0], f2[0]])
        plt.show(block=True)
        y, x = f3[0].get_data()
        self.coordinate[i] = np.vstack([x, y])

    def update_component_map(self, mean):
        cmap = np.ones([self.nx, self.ny]) * -1
        cnt = 0
        for i in (mean.T)[:]:
            x, y = i.astype("int16")
            try:
                cmap[x, y] = cnt
            except:
                pass
            cnt += 1
        return cmap.astype("int16")

    def import_coordinates(self, coordinates, atom_type=None):
        if coordinates is not None:
            self.coordinate.append(coordinates)
        if atom_type is not None:
            self.atom_type = atom_type

    def find_peak(self, th_dist=0, th_inten=0, b_user_confirm=True, confirm_list=None):
        img = self.remove_freq(self.image, 0, 1 / th_dist)
        bitmap = self.local_max(img)
        bitmap = self.apply_threshold(bitmap, self.image, th_inten)
        img = self.apply_threshold(img, self.image, th_inten)
        coor = np.argwhere(bitmap == 1).T.astype(np.float32)
        self.coordinate = []
        for i in range(self.nz):
            id = np.argwhere(coor[0, :] == i).T[0]
            self.coordinate.append(coor[1:, id])
        self.image_processed = img
        if confirm_list is None:
            confirm_list = range(self.nz)
        if b_user_confirm:
            for i in confirm_list:
                self.__interactive_add_remove__(i)

    def fit_gaussian(self, id=0, different_width=False, extend=0, view=False, limit=50):
        n_model = np.size(self.coordinate[id], 1)
        density, map, bg_region = self.__guess_density__(id)
        r = int(density**0.5)
        use_coor = np.argwhere(map == 1).T
        self.image_cropped[id, use_coor[0], use_coor[1]] = self.image[
            id, use_coor[0], use_coor[1]
        ]
        intensity = self.image_cropped[id]
        background = np.sum(self.image[id] * bg_region) / np.sum(bg_region)
        weight = (
            self.image[
                id,
                self.coordinate[id][0, :].astype(np.int64),
                self.coordinate[id][1, :].astype(np.int64),
            ]
            - background
        )
        weight = np.maximum(weight, 0)
        mean = self.coordinate[id]
        sig = np.ones(len(weight)) * r / 2

        params = {
            "sigma": sig[0],
            "weight": weight,
            "pos_x": mean[0],
            "pos_y": mean[1],
            "background": background,
            "r": r,
        }
        optimized_params = self.__update_global__(params, intensity)
        sig = optimized_params["sigma"] * np.ones(len(weight))
        background = optimized_params["background"]

        intensity_fit = calc_fit_outcome(
            self.nx, self.ny, weight, mean[0], mean[1], sig, background, r * 3
        )
        mse = ((intensity_fit - intensity) ** 2).mean()
        weight *= intensity.mean() / intensity_fit.mean()
        component_map = self.update_component_map(mean)

        self.image_fitted[id] = np.copy(intensity_fit)

        step_size = 1
        change_rate = 1
        cnt = 0
        fail_cnt = 0
        different_width_start = False
        finish_postpone = False
        weight_new = np.copy(weight)
        mean_new = np.copy(mean)
        sig_new = np.copy(sig)
        background_new = np.copy(background)
        mse_list = []
        chage_rate_list = []

        while (
            change_rate > 5e-3 and fail_cnt < 5 and cnt < limit and not finish_postpone
        ):
            fitrange = r
            for i in tqdm(range(n_model), desc="Fitting atomic columns"):
                (
                    weight_update,
                    mean_x_update,
                    mean_y_update,
                    sig_update,
                    bg_local,
                    int_update,
                    coorx,
                    coory,
                    b_fit,
                ) = self.__update_local__(
                    i,
                    weight_new,
                    mean_new,
                    sig_new,
                    fitrange,
                    id,
                    component_map,
                    different_width=different_width_start,
                    different_background=False,
                )
                if b_fit:
                    weight_new[i] = weight[i] + (weight_update - weight[i]) * step_size
                    mean_new[0, i] = (
                        mean[0, i] + (mean_x_update - mean[0, i]) * step_size
                    )
                    mean_new[1, i] = (
                        mean[1, i] + (mean_y_update - mean[1, i]) * step_size
                    )
                    sig_new[i] = sig[i] + (sig_update - sig[i]) * step_size

                    self.image_fitted[id, coorx, coory] = int_update + gaussian(
                        coorx,
                        coory,
                        weight_new[i],
                        mean_new[0, i],
                        mean_new[1, i],
                        sig_new[i],
                        0,
                    )
                    self.image_fitted[id] *= map

                    intensity_fit_temp = self.image_fitted[id]
                    mse_temp = ((intensity_fit_temp - intensity) ** 2).mean()
                    if mse_temp - mse > 0:
                        cnt_fit -= 1
                        weight_new[i] = weight[i]
                        mean_new[0, i] = mean[0, i]
                        mean_new[1, i] = mean[1, i]
                        sig_new[i] = sig[i]
                        self.image_fitted[id, coorx, coory] = int_update + gaussian(
                            coorx,
                            coory,
                            weight_new[i],
                            mean_new[0, i],
                            mean_new[1, i],
                            sig_new[i],
                            0,
                        )
                        self.image_fitted[id] *= map

            background_new = background
            intensity_fit = calc_fit_outcome(
                self.nx,
                self.ny,
                weight_new,
                mean_new[0],
                mean_new[1],
                sig_new,
                background_new,
                r * 3,
            )
            mse_new = ((intensity_fit - intensity) ** 2).mean()

            if mse_new - mse > 0:
                step_size /= 2
                print("step size:", step_size)
                fail_cnt += 1
                intensity_fit = calc_fit_outcome(
                    self.nx,
                    self.ny,
                    weight,
                    mean[0],
                    mean[1],
                    sig,
                    background,
                    r * 3,
                )
                self.image_fitted[id, use_coor[0], use_coor[1]] = np.copy(intensity_fit)
            else:
                change_rate = np.abs(mse_new - mse) / np.abs(mse)
                mse = np.copy(mse_new)
                weight = np.copy(weight_new)
                mean = np.copy(mean_new)
                sig = np.copy(sig_new)
                background = np.copy(background_new)
                component_map = self.update_component_map(mean)
                mse_list.append(mse)
                chage_rate_list.append(change_rate)

                self.image_fitted[id] = np.copy(intensity_fit)

                fail_cnt = 0
                cnt += 1

            if change_rate < 1e-2 and different_width and not different_width_start:
                different_width_start = True
                extend1 = extend
                step_size = 1
                fitrange = r * 2
                print("fit sig")
                change_rate = 1e-1  # postpone once

            # if change_rate < 1e-2 and not different_width and not finish_postpone:
            # if change_rate < 1e-2:
            # fitrange = r*2
            # finish_postpone = True
            # change_rate = 2e-3 # postpone once
        if view:
            plt.plot(mse_list)
            plt.xlabel("iteration")
            plt.ylabel("MSE")
            plt.show()
            plt.plot(chage_rate_list)
            plt.xlabel("iteration")
            plt.ylabel("Change rate")
            plt.show()
            fig, axs = plt.subplots(1, 3)
            fig.dpi = 150
            axs[0].imshow(self.image[id])
            axs[0].set_title("Original image")
            axs[1].imshow(self.image_fitted[id])
            axs[1].set_title("Fitting result")
            axs[2].imshow(self.image[id] - self.image_fitted[id])
            axs[2].set_title("Difference")
            # turn off the axis
            axs[0].axis("off")
            axs[1].axis("off")
            axs[2].axis("off")
            plt.show(block=False)
            plt.title("Fitting result")

        self.use_coor = use_coor
        self.gaussian_prmt[id] = {
            "weight": weight,
            "mean": mean,
            "sig": sig,
            "background": background,
        }
        self.image_fitted[id] = calc_fit_outcome(
            self.nx, self.ny, weight, mean[0], mean[1], sig, background, r * 3
        )

    def __update_global__(self, initial_params, intensity):
        # Initial parameters as a dictionary

        # Define a wrapper function for the loss function
        # that accepts a list of parameters and converts it to the expected dictionary format
        def loss_wrapper(param_list):
            updated_params = initial_params.copy()
            updated_params["sigma"], updated_params["background"] = param_list
            return loss(intensity, updated_params)

        # Initial guess list for fmin
        p0 = [initial_params["sigma"], initial_params["background"]]

        # Use fmin with the wrapper function
        optimized_params_list = fmin(loss_wrapper, p0)

        # Update the original parameters dictionary with the optimized values
        optimized_params = initial_params.copy()
        optimized_params["sigma"] = optimized_params_list[0]
        optimized_params["background"] = optimized_params_list[1]

        return optimized_params

    def __update_local__(
        self,
        i,
        weight,
        mean,
        sig,
        r,
        id,
        component_map,
        different_width=False,
        different_background=False,
    ):
        i_l, i_r = np.maximum(mean[0, i] - r, 0).astype(int), np.minimum(
            mean[0, i] + r + 1, self.nx
        ).astype(int)
        i_u, i_d = np.maximum(mean[1, i] - r, 0).astype(int), np.minimum(
            mean[1, i] + r + 1, self.ny
        ).astype(int)

        # Flatten the intensity and fitting arrays
        intensity = self.image_cropped[id][i_l:i_r, i_u:i_d].ravel()
        intensity_fit = self.image_fitted[id][i_l:i_r, i_u:i_d].ravel()

        # Create meshgrid and flatten
        xv, yv = np.meshgrid(np.arange(i_l, i_r), np.arange(i_u, i_d), indexing="ij")
        xv, yv = xv.flatten(), yv.flatten()

        # Initialize background to 0
        bg = 0

        # Construct fitting_input based on conditions
        fitting_input = np.vstack([xv, yv])
        if different_width:
            if not different_background:
                fitting_input = np.vstack([fitting_input, np.full(xv.size, bg)])
        else:
            sig_value = (
                np.full(xv.size, sig[0])
                if different_background
                else np.full(xv.size, sig[i])
            )
            fitting_input = np.vstack([fitting_input, sig_value, np.full(xv.size, bg)])

        # remove contribution from neighbors
        intensity_fit -= gaussian(
            fitting_input[0],
            fitting_input[1],
            weight[i],
            mean[0, i],
            mean[1, i],
            sig[i],
            bg,
        )
        intensity_fit_target = np.copy(intensity_fit)
        intensity -= intensity_fit

        # Prepare initial parameters and bounds based on conditions
        p0 = [weight[i], mean[0, i], mean[1, i]] + ([sig[i]] if different_width else [])
        bounds_lower = [0, i_l, i_u] + ([0] if different_width else [])
        bounds_upper = [np.inf, i_r, i_d] + ([np.inf] if different_width else [])
        bounds = (bounds_lower, bounds_upper)

        # Curve fitting
        try:
            fitting_func = (
                gaussian_sum_different_sigma
                if different_width
                else gaussian_sum_same_sigma
            )
            optimized_params, _ = curve_fit(
                fitting_func, fitting_input, intensity, p0, bounds=bounds
            )
            if different_width:
                result = list(optimized_params) + [
                    bg,
                    intensity_fit_target,
                    fitting_input[0, :].astype("int16"),
                    fitting_input[1, :].astype("int16"),
                    True,
                ]
            else:
                result = list(optimized_params) + [
                    sig[i],
                    bg,
                    intensity_fit_target,
                    fitting_input[0, :].astype("int16"),
                    fitting_input[1, :].astype("int16"),
                    True,
                ]
        except RuntimeError:
            print(
                f"Fitting for component {i} Failed! Coordinate: ({mean[0, i]}, {mean[1, i]})."
            )
            result = (
                weight[i],
                mean[0, i],
                mean[1, i],
                sig[i],
                bg,
                intensity_fit_target,
                fitting_input[0, :].astype("int16"),
                fitting_input[1, :].astype("int16"),
                False,
            )
        return result

    def plot_image(self, id=0):
        fig, axs = plt.subplots(2, 2)
        fig.dpi = 150
        axs[0, 0].imshow(self.image[id])
        axs[0, 1].imshow(self.image_processed[id])
        axs[1, 0].plot(
            self.coordinate[id][1], self.coordinate[id][0], ".k", markersize=1
        )
        axs[1, 0].set_aspect("equal")
        axs[1, 0].set(xlim=(0, self.ny), ylim=(0, self.nx))
        axs[1, 0].invert_yaxis()
        axs[1, 1].imshow(
            self.image_fitted[id], vmin=self.image[id].min(), vmax=self.image[id].max()
        )
        plt.show(block=False)
        plt.pause(1)


################################################################################################################################
class AddCoordinate:
    def __init__(self, line, init_coordinate, extra_line):
        self.line = line
        # self.ax_x, self.ax_y = line.shape
        self.y, self.x = init_coordinate
        self.cid = self.line.figure.canvas.mpl_connect("button_press_event", self)
        self.extra_line = extra_line

    def __call__(self, event):
        # print(event.xdata, ', ', event.ydata)
        # print(self.x.shape)
        if event.button == 1:
            self.x = np.append(self.x, event.xdata)
            self.y = np.append(self.y, event.ydata)
            self.line.set_data(self.x, self.y)
            self.line.figure.canvas.draw()
        if event.button == 3:
            dist = (self.x - event.xdata) ** 2 + (self.y - event.ydata) ** 2
            id_min = np.argmin(dist)
            # print(self.x[id_min], ', ', self.y[id_min] )
            self.x = np.delete(self.x, id_min)
            self.y = np.delete(self.y, id_min)
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.draw()
        for el in self.extra_line:
            # el.clear()
            el.set_data(self.x, self.y)
            el.figure.canvas.draw()


################################################################################################################################


class GaussianMixtureModel:
    def __init__(self, scs: np.ndarray, electron_per_px=None):
        self.scs = scs
        self.dose = electron_per_px
        self.result = {}
        self.val: np.ndarray
        self.minmax: np.ndarray
        self.init_mean: np.ndarray
        self.curve = None
        self.curve_prmt = None

    def initCondition(
        self,
        n_component,
        use_scs_channel,
        metric,
        score_method,
        init_method,
        lim_rate,
        lim_ite,
        given_weight,
        given_mean,
        given_width,
        fit_step_size,
        constraint,
    ):
        # prepare data
        if isinstance(n_component, int):
            self.n_component_list = np.arange(1, n_component + 1)
        elif isinstance(n_component, list):
            if given_mean is None:
                n_component[0] = 1
            self.n_component_list = np.arange(n_component[0], n_component[-1] + 1)
        score_method.append("nllh")
        self.score_method = score_method
        self.lim_rate = lim_rate
        self.lim_ite = lim_ite
        if use_scs_channel is None:
            if np.size(self.scs, 1) == 1:
                use_scs_channel = [0]
            elif np.size(self.scs, 1) == 2:
                use_scs_channel = [0, 1]
            else:
                print("only support up to 2D GMM, only use first channel for now\n")
                use_scs_channel = [0]
        elif isinstance(use_scs_channel, int):
            use_scs_channel = [use_scs_channel]
        self.channel = use_scs_channel
        self.n_dim = len(self.channel)
        self.val = self.scs[:, self.channel]
        self.minmax = np.array([self.val.min(0), self.val.max(0)])
        self.metric = metric
        self.given_weight = given_weight
        self.given_mean = given_mean
        self.given_width = given_width
        if np.size(self.val, 1) == 2:
            self.fit_method = self.polyCurve_5
            self.curve_prmt, _ = curve_fit(
                self.fit_method, self.val[:, 0], self.val[:, 1]
            )
        self.init_method = init_method
        self.step = fit_step_size
        self.constraint = constraint

    def GMM(
        self,
        name: str,
        n_component,
        use_scs_channel=None,
        metric="nllh",
        score_method=["icl"],
        init_method="middle",
        lim_rate=1e-5,
        lim_ite=1e5,
        given_weight=None,
        given_mean=None,
        given_width=None,
        fit_step_size=[1, [1, 1], [1, 1]],
        constraint=[],
    ):
        # constraint = ['uni_width', 'no_cov', '45deg', 'dose_width']

        self.initCondition(
            n_component,
            use_scs_channel,
            metric,
            score_method,
            init_method,
            lim_rate,
            lim_ite,
            given_weight,
            given_mean,
            given_width,
            fit_step_size,
            constraint,
        )
        gmm_result = self.initResultDict(
            self.n_component_list, score_method, len(self.channel) == 2
        )

        wmw = [np.array([]), np.array([[None]]), np.array([[]])]

        for n in tqdm(self.n_component_list):
            wmw, score = self.EmOptimization(n, last_mean=wmw[1])
            gmm_result["weight"].append(wmw[0])
            gmm_result["mean"].append(wmw[1])  # the variance of the gaussian
            gmm_result["width"].append(wmw[2])
            for key in score:
                gmm_result["score"][key].append(score[key])

        self.result[name] = GmmResult(
            name,
            gmm_result["weight"],
            gmm_result["mean"],
            gmm_result["width"],
            gmm_result["score"],
            np.size(self.scs, 1),
            self.val,
            self.curve_prmt,
        )

    def EmOptimization(self, n_component, last_mean):
        mean_list = self.initMean(self.init_method, last_mean, n_component)
        if self.n_dim == 2 and not np.size(mean_list[0], 1) == 2:
            mean_list = self.addChannel(mean_list, self.fit_method, self.curve_prmt)
        wmw_list = []
        score_list = {key: [] for key in self.score_method}
        for mean in mean_list:
            weight = self.initWeight(n_component)
            width = self.initWidth(n_component)
            wmw, score = self.EM(weight, mean, width)
            wmw_list.append(wmw)
            for key in score:
                score_list[key].append(score[key])
        idx = np.argmin(score_list[self.metric])
        min_score = {key: score_list[key][idx] for key in score}
        return wmw_list[idx], min_score

    def initWeight(self, n_component):
        if self.given_weight is None:
            weight = np.ones(n_component) / n_component
        else:
            if isinstance(self.given_weight, list):
                weight = self.given_weight[n_component - 1]
            else:
                weight = self.given_weight[:n_component]
        return weight

    def initWidth(self, n_component):
        width_temp = ((self.minmax[1] - self.minmax[0]) / (2 * n_component)) ** 2
        width_temp = np.expand_dims(
            width_temp, axis=0
        )  # give common axis for component dimension
        if self.given_width is None:
            return width_temp
        else:
            if isinstance(self.given_width, list):
                width = self.given_width[n_component - 1]
            else:
                width = self.given_width[:n_component]
            if width_temp.shape[2] == 2:
                if width.shape[0] > 1:
                    width_temp = np.repeat(width_temp, n_component, axis=0)
                if width.shape[2] < 2:
                    width = np.concatenate([width, [width_temp[..., -1]]], axis=-1)
            return width

    def initMean(self, init_method, last_mean, n_component):
        minmax = [self.minmax[0][0], self.minmax[1][0]]
        if n_component == 1 and self.init_method != "initvalue":
            mean = np.zeros((1, 1))
            mean[0, 0] = (minmax[0] + minmax[1]) / 2
            mean_list = [mean]
        else:
            last_mean = np.expand_dims(
                last_mean[:, 0], -1
            )  # only use the first channel for initialization

            # mehtod 1
            if init_method == "equionce":
                mean_list = [
                    np.expand_dims(
                        np.linspace(
                            minmax[0], minmax[1], n_component + 1, endpoint=False
                        )[1:],
                        -1,
                    )
                ]

            # mehtod 2
            if init_method == "equimul":
                repeat = 20
                mean_0 = np.expand_dims(
                    np.linspace(minmax[0], minmax[1], n_component + 1, endpoint=False)[
                        1:
                    ],
                    -1,
                )
                delta = (
                    (mean_0[1] - mean_0[0])
                    * (np.random.rand(repeat, mean_0.shape[0], mean_0.shape[1]) - 0.5)
                    / 2
                )
                mean = delta + mean_0
                mean[mean < minmax[0]] = minmax[0]
                mean[mean > minmax[1]] = minmax[1]
                mean_list = list(mean)

            # mehtod 3
            if init_method == "middle":
                points = np.insert(last_mean, (0, n_component - 1), minmax)
                mean_list = []
                for n in range(n_component):
                    new_point = (points[n] + points[n + 1]) / 2
                    mean_list.append(np.insert(last_mean, n, new_point, axis=0))

            # mehtod 4
            if init_method == "finegrid":
                points = np.linspace(
                    minmax[0], minmax[1], self.n_component_list[-1] + 1, endpoint=False
                )[1:]
                mean_list = []
                for p in points:
                    mean_list.append(np.sort(np.insert(last_mean, 0, p, axis=0)))

            # method 5
            if init_method == "initvalue":
                if isinstance(self.given_mean, list):
                    mean_list = [self.given_mean[n_component - 1, self.channel]]
                else:
                    mean_list = [self.given_mean[:n_component, self.channel]]
        return mean_list

    def EM(self, weight, mean, width):
        g = GaussianComponents(weight, mean, width, self.val, self.dose)
        if g.b_fail:
            return [weight, mean, width], self.failedScore()
        llh = self.logLikelihood(g.ca)
        rate = 1
        cnt = 0
        # g.preMSTep(self.step, self.constraint)
        while (rate > self.lim_rate) and (cnt < self.lim_ite):
            # g.EStep()
            g.MStep(self.step, self.constraint)
            if self.meanCoincide(g.mean) or g.b_fail:
                break
            llh_new = self.logLikelihood(g.ca)
            rate = abs(llh_new - llh) / abs(llh)
            llh = llh_new
            cnt += 1
        if cnt == self.lim_ite:
            print("fitting did not converge\n")
        weight, mean, width = [g.weight, g.mean, g.var]
        sort_idx = np.argsort(mean[:, 0])
        mean = np.take_along_axis(mean, np.expand_dims(sort_idx, -1), axis=0)
        weight = np.take_along_axis(weight, sort_idx, axis=0)
        score = self.calculateScore(g.tau, llh)
        return [weight, mean, width], score

    @staticmethod
    def meanCoincide(mean):
        cri = 1e-3
        diff = mean[1:] - mean[:-1]
        if ((diff**2).sum(1) ** 0.5 < cri).any():
            # print('mean coincide')
            return True
        else:
            return False

    @staticmethod
    def logLikelihood(array):
        llh = np.log(array.sum(0)).sum()
        return llh

    @staticmethod
    def componentArray(weight, mean, width, val):
        components = weight * np.prod(width * 2 * np.pi, axis=-1) ** -(1 / 2)
        if np.size(mean, 1) == 2:
            distance2 = np.sum(
                (val - np.expand_dims(mean, 1)) ** 2 / np.expand_dims(width, 1) / 2, -1
            )
        else:
            distance2 = np.squeeze(
                (val - np.expand_dims(mean, 1)) ** 2 / np.expand_dims(width, 1) / 2
            )
        distance_term = np.exp(-distance2)
        component_array = components.reshape((len(components), 1)) * distance_term
        return component_array

    def calculateScore(self, tau, llh):
        penalty = 2
        n_component, n_val = np.shape(tau)
        ## weight
        n_para = (self.step[0] != 0) * (n_component - 1)
        ## mean
        n_para_mean = n_component * self.n_dim
        n_para += n_para_mean
        ## width
        n_para_width = [1, n_component]
        if "uni_width" in self.constraint:
            n_para_width[1] = 1
        if self.n_dim == 2:
            n_para_width[0] = 3
            if ("45deg" in self.constraint) or ("no_cov" in self.constraint):
                n_para_width[0] = 2
        n_para += n_para_width[0] * n_para_width[1]

        t = tau * safe_ln(tau)
        t[tau == 0] = 0
        en = -1 * np.sum(t)

        score = {}
        for key in self.score_method:
            if key == "aic":
                score[key] = -2 * llh / self.n_dim + 2 * n_para
            if key == "gic":
                score[key] = -2 * llh / self.n_dim + penalty * n_para
            if key == "bic":
                score[key] = -2 * llh / self.n_dim + n_para * np.log(n_para)
            if key == "clc":
                score[key] = -2 * llh / self.n_dim + 2 * en
            if key == "awe":
                score[key] = (
                    -2 * llh / self.n_dim
                    + 2 * en
                    + 2 * n_para * (3 / 2 + np.log(n_val))
                )
            if key == "icl":
                score[key] = -2 * llh / self.n_dim + 2 * en + n_para * np.log(n_val)
            if key == "nllh":
                score[key] = -1 * llh
            if key == "en":
                score[key] = en
        return score

    def failedScore(self):
        score = {}
        for key in self.score_method:
            score[key] = np.inf
        return score

    def applyConstraint(self, weight, mean, width):
        # if 'uni_width' in self.constraint:
        #     width = np.expand_dims(width.mean(0), axis=0)
        if ("no_cov" in self.constraint) and self.n_dim == 2:
            mask = np.zeros(width.shape)
            for n in range(self.n_dim):
                mask[:, n, n] = 1
            width = width * mask
        if ("45deg" in self.constraint) and self.n_dim == 2:
            val = np.max([width[:, 0, 0], width[:, 1, 1]])
            # val = np.abs(width).max()
            for n in range(0, self.n_dim):
                width[:, n, n] = val
        return weight, mean, width

    @staticmethod
    def addChannel(list, method, prmt):
        return [np.concatenate((val, method(val, *prmt)), axis=-1) for val in list]

    @staticmethod
    def initResultDict(n_component_list: int, score_method: list, b_2d: bool):
        n_cases = len(n_component_list)
        score_dict = {key: [] for key in score_method}
        gmm_result = {
            "weight": [],
            "mean": [],
            "width": [],
            "score": score_dict,
            "scsidx": [],
            "case": n_component_list,
        }
        return gmm_result

    @staticmethod
    def expCurve(x, a, b, c):
        return x**a * b + c

    @staticmethod
    def polyCurve_1(x, a, b):
        return x * a + b

    @staticmethod
    def polyCurve_2(x, a, b, c):
        return x**2 * a + x * b + c

    @staticmethod
    def polyCurve_3(x, a, b, c, d):
        return x**3 * a + x**2 * b + x * c + d

    @staticmethod
    def polyCurve_4(x, a, b, c, d, e):
        return x**4 * a + x**3 * b + x**2 * c + x * d + e

    @staticmethod
    def polyCurve_5(x, a, b, c, d, e, f):
        return x**5 * a + x**4 * b + x**3 * c + x**2 * d + x * e + f

    @staticmethod
    def polyCurve_6(x, a, b, c, d, e, f, g):
        return x**6 * a + x**5 * b + x**4 * c + x**3 * d + x**2 * e + x * f + g


class GmmResult:
    def __init__(
        self,
        name: str,
        weight: list,
        mean: list,
        width: list,
        score: dict,
        ndim: int,
        val: np.ndarray,
        curve=None,
    ):
        self.name = name
        self.weight = weight
        self.mean = mean
        self.width = width
        self.score = score
        self.curve = curve
        self.ndim = ndim
        self.val = val

    def idxComponentOfScs(self, id):
        # point each scs to a specific component
        g = GaussianComponents(self.weight[id], self.mean[id], self.width[id])
        ca = g.componentArray(self.val)
        return np.argmax(ca, 0)

    def idxScsOfComponent(self, id):
        # list scs under each component
        idx_c = self.idxComponentOfScs(id)
        idx_s = []
        for c in range(id + 1):
            idx_s.append(np.argwhere(idx_c == c))
        return idx_s


class GaussianComponents:
    def __init__(self, weight, mean, var, val, dose=None):
        self.weight = weight
        self.mean = mean
        self.var = var
        self.val = val
        self.n_comp = self.weight.shape
        self.n_val, self.n_dim = self.val.shape
        self.dose = dose
        self.b_fail = False
        self.componentArray()
        self._tau
        self.tau_sum
        self.tau_ex

    def componentArray(self):
        #                   ORIGINAL AXIS         ->     EXPANDED AXIS
        # weight:           (component)           ->  (component, channel)
        # mean:             (component, channel)  ->  (component, data, channel)
        # var (same width): (1, channel)          ->  (1, data, channel)
        # var (diff width): (component, channel)  ->  (component, data, channel)
        # val:              (data, channel)
        # c:                (component, data)
        var = np.expand_dims(self.var, axis=1)
        mean = np.expand_dims(self.mean, 1)
        weight = np.expand_dims(self.weight, 1)
        ca = self._ca(var, mean, weight, self.val)
        self.EStep(ca)
        if not self.b_fail:
            self.ca = ca

    @staticmethod
    def _ca(var, mean, weight, val):
        n_dim = mean.shape[-1]
        dis = (val - mean) ** 2
        if (var == 0).any():
            print("h")
        ca = (
            (2 * np.pi) ** (-n_dim / 2)
            * np.prod(var, -1) ** (-1 / 2)
            * np.exp(-1 / 2 * (dis / var).sum(-1))
        )
        ca *= weight
        return ca

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self.tau_sum = tau.sum(1) + np.finfo("double").eps
        self.tau_ex = np.expand_dims(tau, -1)
        self.tau_ex_sum = self.tau_ex.sum(1) + np.finfo("double").eps

    def EStep(self, ca):
        self.tau = ca / np.sum(ca, 0)
        if (self.tau.sum(1) < 1).any():
            self.b_fail = True

    def preMSTep(self, step, constraint):
        self.updateWeight(step[0])
        self.componentArray()
        self.updateVariance(step[2], constraint)
        self.componentArray()

    def MStep(self, step, constraint):
        self.updateWeight(step[0])
        self.updateMean(step[1])
        self.updateVariance(step[2], constraint)
        self.componentArray()

    def updateWeight(self, step):
        new_weight = self.tau_sum / self.n_val
        self.weight = (new_weight - self.weight) * step + self.weight
        self.weight[-1] = 1 - np.sum(self.weight[:-1])

    def updateMean(self, step):
        new_mean = (self.tau_ex * self.val).sum(1) / self.tau_ex_sum
        self.mean = (new_mean - self.mean) * step[: self.n_dim] + self.mean

    def updateVariance(self, step, constraint):
        mean = np.expand_dims(self.mean, 1)
        var = np.expand_dims(self.var, 1)
        dis = (self.val - mean) ** 2
        if "uni_width" in constraint:
            if "dose_width" in constraint:
                var_dose = mean / self.dose
                var_indi = ((self.tau_ex * dis - var_dose) / var**2).sum((0, 1)) / (
                    self.tau_ex / var**2
                ).sum((0, 1))
                var_indi[var_indi < 0] = 0
                new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
            elif "dose_width_simplified" in constraint:
                var_dose = mean / self.dose
                var_indi = (self.tau_ex * dis - var_dose).sum((0, 1)) / self.n_val
                var_indi[var_indi < 0] = 0
                new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
            elif "dose_width_fit" in constraint:
                var = np.expand_dims(self.var, axis=1)
                mean = np.expand_dims(self.mean, 1)
                weight = np.expand_dims(self.weight, 1)
                var_dose = mean / self.dose
                args = (var_dose, mean, weight, self.val)
                var_indi = (var - var_dose).mean((0, 1))
                var_indi[var_indi < np.finfo("double").eps] = np.finfo("double").eps
                result = minimize(
                    self.__fit_dose_indi_var__, self.rev_rectifier(var_indi), args=args
                )
                if result.success:
                    var_indi = self.rectifier(result.x)
                    new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
                else:
                    # print('cannot find solution')
                    new_var = self.var
            else:
                new_var = np.expand_dims(
                    (self.tau_ex * dis).sum((0, 1)) / self.n_val, 0
                )
        else:
            new_var = (self.tau_ex * dis).sum(1) / self.tau_ex_sum
        self.var = (new_var - self.var) * step[: self.n_dim] + self.var

    @staticmethod
    def rectifier(val):
        return np.exp(val)

    @staticmethod
    def rev_rectifier(val):
        return np.log(val)

    def __fit_dose_indi_var__(self, var_indi, *args):
        # args: var_dose, mean, weight, val
        # fitting by minimizing negative log-likelihood
        var_dose, mean, weight, val = args
        nllh = -np.log(
            self._ca(self.rectifier(var_indi) + var_dose, mean, weight, val).sum(0)
        ).sum()
        return nllh
