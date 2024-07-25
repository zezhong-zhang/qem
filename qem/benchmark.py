import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

import qem
from qem.image_fitting import ImageModelFitting


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(
            f"Method {func.__name__!r} executed in {(end_time - start_time):.4f} seconds"
        )
        return result

    return wrapper


def get_coordinates(StatSTEM):
    # Check for 'coordinates' or 'Coordinates' directly
    if "coordinates" in StatSTEM:
        coordinates = StatSTEM["coordinates"]
    elif "Coordinates" in StatSTEM:
        coordinates = StatSTEM["Coordinates"]
    # If direct coordinates are not found, check for 'BetaX' and 'BetaY'
    elif "BetaX" in StatSTEM and "BetaY" in StatSTEM:
        BetaX = StatSTEM["BetaX"]
        BetaY = StatSTEM["BetaY"]
        coordinates = np.array([BetaX, BetaY]).T
    else:
        raise ValueError("Coordinate keys not found in inputStatSTEM dictionary.")
    return coordinates


def get_scs(StatSTEM):
    if "volumes" in StatSTEM:
        scs = StatSTEM["volumes"]
    elif "Volumes" in StatSTEM:
        scs = StatSTEM["Volumes"]
    else:
        raise ValueError("Volume keys not found in inputStatSTEM dictionary.")
    return scs


class Benchmark:
    def __init__(self, filepath):
        legacyStatSTEM = qem.io.read_legacyInputStatSTEM(filepath)
        try:
            if "dx" in legacyStatSTEM.keys():
                self.dx = legacyStatSTEM["dx"]
            elif "dx" in legacyStatSTEM["input"].keys():
                self.dx = legacyStatSTEM["input"]["dx"]
            if "input" in legacyStatSTEM.keys():
                inputStatSTEM = legacyStatSTEM["input"]
                self.input_coordinates = get_coordinates(inputStatSTEM)
                self.image = inputStatSTEM["obs"]
            if "output" in legacyStatSTEM.keys():
                outputStatSTEM = legacyStatSTEM["output"]
                self.output_coordinates = get_coordinates(outputStatSTEM)
                if "model" in outputStatSTEM.keys():
                    self.model_statstem = outputStatSTEM["model"]
                if (
                    "volumes" in outputStatSTEM.keys()
                    or "Volumes" in outputStatSTEM.keys()
                ):
                    self.scs_statstem = get_scs(outputStatSTEM)
            if "obs" in legacyStatSTEM.keys():
                self.image = legacyStatSTEM["obs"]
            if "coordinates" in legacyStatSTEM.keys():
                self.input_coordinates = legacyStatSTEM["coordinates"]
            if "model" in legacyStatSTEM.keys():
                self.model_statstem = legacyStatSTEM["model"]
        except:
            raise ValueError(
                "InputStatSTEM dictionary does not have correct keys in the input file."
            )

    @time_it
    def refine(self, atom_size = 0.7, guess_radius = False, tol = 1e-2, maxiter = 50, step_size = 1e-2, num_epoch = 10, batch_size = 1000, verbose = False, plot = True) -> None:
        model=ImageModelFitting(self.image, dx=self.dx)
        model.coordinates=self.input_coordinates/self.dx
        params = model.init_params(atom_size=atom_size, guess_radius=guess_radius)
        params = model.fit_random_batch(params, tol=tol, maxiter=maxiter, step_size=step_size, num_epoch=num_epoch, batch_size=batch_size, verbose=verbose, plot=plot)
        self.qem = model
        self.model_qem = model.model
        self.scs_qem = model.volume
        self.params_qem = params
        self.qem.voronoi_integration(plot=True)
        self.scs_voronoi = self.qem.voronoi_volume


    def compare_scs_voronoi(self, folder_path=None, file_path=None, save=False):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        im = plt.scatter(self.qem.params['pos_x'], self.qem.params['pos_y'], s=1, c=self.scs_qem, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("QEM refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1, 3, 2)
        im = plt.scatter(self.qem.params['pos_x'], self.qem.params['pos_y'], s=1, c=self.scs_voronoi, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Voronoi refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        im = plt.scatter(self.qem.params['pos_x'], self.qem.params['pos_y'], s=1, c=self.scs_voronoi-self.scs_qem, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("difference refined scs ($\AA^2$)")
        plt.clim(-self.scs_qem.mean() / 10, self.scs_qem.mean() / 10)
        plt.tight_layout()
        if save:
            if file_path is None:
                file_path = "voronoi_scs.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)


    def compare_residual(
        self, mode="both", folder_path=None, file_path=None, save=False
    ):
        image = self.image
        if mode == "StatSTEM":
            plt.subplots(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("image")
            plt.subplot(1, 3, 2)
            im = plt.imshow(self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("StatSTEM prediction")
            plt.subplot(1, 3, 3)
            im = plt.imshow(image - self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 100, image.mean() / 100)
            plt.title("difference")
            plt.tight_layout()
        if mode == "QEM":
            plt.subplots(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("image")
            plt.subplot(1, 3, 2)
            im = plt.imshow(self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.title("QEM prediction")
            plt.subplot(1, 3, 3)
            im = plt.imshow(image - self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("difference")
            plt.tight_layout()
            plt.clim(-image.mean() / 100, image.mean() / 100)
        if mode == "both":
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 3, 1)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Input Image")

            plt.subplot(2, 3, 2)
            im = plt.imshow(self.model_qem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(image.min(), image.max())
            plt.tight_layout()
            plt.title("QEM Model")

            plt.subplot(2, 3, 3)
            diff = image - self.model_qem
            im = plt.imshow(diff)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 10, image.mean() / 10)
            plt.tight_layout()
            plt.title("Residuals")

            plt.subplot(2, 3, 4)
            im = plt.imshow(image)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Input Image")

            plt.subplot(2, 3, 5)
            im = plt.imshow(self.model_statstem)
            plt.clim(image.min(), image.max())
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.title("Legacy StatSTEM Model")

            plt.subplot(2, 3, 6)
            im = plt.imshow(image - self.model_statstem)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.clim(-image.mean() / 10, image.mean() / 10)
            plt.tight_layout()
            plt.title("Residuals")
        if save:
            if file_path is None:
                file_path = "residuals.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def compare_scs_map(self, folder_path=None, file_path=None, save=False):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        pos_x = self.params_qem['pos_x']*self.dx
        pos_y = self.params_qem['pos_y']*self.dx
        pos_x_statstem = self.output_coordinates[:, 0]
        pos_y_statstem = self.output_coordinates[:, 1]
        index_statstem_in_qem = np.array(
            [
                np.argmin(
                    np.sqrt((pos_x_statstem - x) ** 2 + (pos_y_statstem - y) ** 2)
                )
                for x, y in zip(pos_x, pos_y)
            ]
        )
        pos_x_statstem = pos_x_statstem[index_statstem_in_qem]
        pos_y_statstem = pos_y_statstem[index_statstem_in_qem]
        volume_statstem = volume_statstem[index_statstem_in_qem]
        

        plt.subplots(figsize=(15,5))
        plt.subplot(1,3,1)
        im = plt.scatter(pos_x, pos_y, c=volume_qem, s=2)
        # make aspect ratio equal
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"QEM refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.subplot(1,3,2)
        im = plt.scatter(pos_x_statstem,pos_y_statstem, c=volume_statstem, s=2)
        plt.gca().invert_yaxis()
        plt.clim(volume_qem.min(), volume_qem.max())
        # make aspect ratio equal
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"Matlab StatSTEM refined scs ($\AA^2$)")
        plt.tight_layout()

        plt.subplot(1,3,3)
        im = plt.scatter(pos_x, pos_y, c=volume_statstem-volume_qem, s=2)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(r"difference refined scs ($\AA^2$)")
        plt.tight_layout()
        plt.clim(-volume_qem.mean() / 10, volume_qem.mean() / 10)
        if save:
            if file_path is None:
                file_path = "scs_map.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def compare_scs_histogram(self, folder_path=None, file_path=None, save=False):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        # remove outliners of volume_statstem
        data = volume_statstem.reshape(-1, 1)  # Reshape for compatibility with sklearn
        clf = IsolationForest(
            contamination=0.01
        )  # Estimate of the contamination of the data
        preds = clf.fit_predict(data)
        outliers = data[preds == -1].reshape(-1)
        # only keep the outliers that larger than the mean
        outliers = outliers[outliers > np.mean(volume_statstem)]
        # remove outliers from volume_statstem
        volume_statstem = volume_statstem[
            np.isin(volume_statstem, outliers, invert=True)
        ]
        volume_statstem = volume_statstem[volume_statstem > 0]  # remove negative values
        plt.figure(figsize=(6, 6))
        plt.hist(volume_qem, bins=100, alpha=0.5, label="QEM", density=True)
        plt.hist(volume_statstem, bins=100, alpha=0.5, label="StatSTEM", density=True)
        plt.xlabel("scs ($\AA^2$)")
        plt.ylabel("frequency")
        plt.legend()
        plt.title("Histogram of scs")
        if save:
            if file_path is None:
                file_path = "scs_histogram.png"
            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, file_path)
            plt.savefig(full_path, dpi=300)

    def scs_error(self, relative = True):
        volume_qem = self.scs_qem
        volume_statstem = self.scs_statstem
        if volume_qem.shape != volume_statstem.shape:
            pos_x = self.params_qem['pos_x']*self.dx
            pos_y = self.params_qem['pos_y']*self.dx
            pos_x_statstem = self.output_coordinates[:, 0]
            pos_y_statstem = self.output_coordinates[:, 1]
            index_statstem_in_qem = np.array(
                [
                    np.argmin(
                        np.sqrt((pos_x_statstem - x) ** 2 + (pos_y_statstem - y) ** 2)
                    )
                    for x, y in zip(pos_x, pos_y)
                ]
            )
            volume_statstem = volume_statstem[index_statstem_in_qem]
        mask = (volume_statstem>np.percentile(volume_statstem, 0.1)) & (volume_statstem<np.percentile(volume_statstem, 99.9))
        if relative:
            error = (volume_statstem-volume_qem)/volume_qem
        else:
            error = volume_statstem-volume_qem
        return error[mask].mean(), error[mask].std()
