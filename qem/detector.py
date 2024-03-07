import numpy as np
import matplotlib.pyplot as plt
from cv2 import GaussianBlur, moments
from skimage import segmentation
from .DM import dm_load
from skimage.feature import canny
from scipy import ndimage as ndi
# from numba import jit, njit, prange


class Detector:
    def __init__(self, array) -> None:
        self.detector = array
        self.detector_smooth = GaussianBlur(array, (5, 5), sigmaX=2, sigmaY=2)
        self.detector_normalised = np.zeros_like(array)

    def plot_orginal(self):
        plt.imshow(self.detector, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

    def detect_black_line(self):
        data = self.detector
        num_zeros_col = np.sum(data == 0, axis=0)
        index = num_zeros_col.argmax(axis=0)
        plt.figure()
        plt.plot(num_zeros_col)
        plt.scatter(index, num_zeros_col[index], color="r", label="Dark line")
        plt.scatter(
            index + 1, num_zeros_col[index + 1], color="b", label="Right neighbour"
        )
        plt.scatter(
            index - 1, num_zeros_col[index - 1], color="g", label="Left neighbour"
        )
        plt.title("Black line detection")
        plt.ylabel("Number of zeros")
        plt.xlabel("Column")
        plt.legend()
        plt.show()
        return index

    def detect_white_line(self):
        data = self.detector
        col_sum = np.sum(data, axis=0)
        index = col_sum.argmax(axis=0)
        plt.figure()
        plt.plot(col_sum)
        plt.scatter(index, col_sum[index], color="r", label="White line")
        plt.scatter(
            index - 1, col_sum[index - 1], color="b", label="Left neighbour (dark line)"
        )
        plt.scatter(
            index - 2, col_sum[index - 2], color="g", label="Next left neighbour"
        )
        plt.title("White line detection")
        plt.ylabel("Sum of intensity")
        plt.xlabel("Column")
        plt.legend()
        plt.show()
        return index

    def remove_line_defect(self, data=None, update=False):
        if data is None:
            data = self.detector.copy()
        black_idx = self.detect_black_line()
        plt.figure()
        # avg = (data[:,black_idx-1] + data[:,black_idx] + data[:,black_idx+])/3
        avg = (data[:, black_idx - 2] + data[:, black_idx + 2]) / 2
        data[:, black_idx - 1] = avg
        data[:, black_idx] = avg
        data[:, black_idx + 1] = avg
        plt.plot(np.sum(data, axis=0))
        plt.title("Fixed line defect")
        plt.ylabel("Sum of intensity")
        plt.xlabel("Column")
        plt.show()

        if update:
            # if update is True, update the detector and detector_smooth
            # only do that once, otherwise it will introduce artifacts
            self.detector = data
            self.detector_smooth = GaussianBlur(data, (5, 5), sigmaX=2, sigmaY=2)
        return data

    def remove_hot_pixels(self, data=None, sigma=30, update=False):
        if data is None:
            data = self.remove_line_defect()
        smoothed_data = GaussianBlur(data, (5, 5), sigmaX=5, sigmaY=5)
        difference_image = np.abs(data - smoothed_data)
        threshold = np.mean(difference_image) + sigma * np.std(difference_image)
        hot_pixels = np.where(difference_image > threshold)
        cleaned_data = data.copy()
        cleaned_data[hot_pixels] = smoothed_data[hot_pixels]

        # Display basic information about hot pixels
        num_hot_pixels = len(hot_pixels[0])
        print(
            f"Number of hot pixels: {num_hot_pixels}, orginal-gaussian smooth difference threshold: {threshold}"
        )

        # Visualizing original, difference, and cleaned data
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()

        axes[0].imshow(data, cmap="gray")
        axes[0].set_title("Original Data (after line defect removal)")
        axes[0].axis("off")

        axes[1].imshow(difference_image, cmap="gray")
        axes[1].plot(hot_pixels[1], hot_pixels[0], "ro", markersize=0.5)
        axes[1].set_title("Difference Image + Hot Pixels (in red)")
        axes[1].axis("off")

        axes[2].imshow(cleaned_data, cmap="gray")
        axes[2].set_title("Cleaned Data")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        if update:
            self.detector = cleaned_data
            self.detector_smooth = GaussianBlur(
                cleaned_data, (5, 5), sigmaX=2, sigmaY=2
            )

        return cleaned_data

    def relative_threshold(self, percentile=25):
        # Determine the value corresponding to the given percentile
        threshold_value = np.percentile(self.detector_smooth, 100 - percentile)
        # Generate binary mask
        binary_mask = self.detector_smooth > threshold_value
        plt.imshow(binary_mask, cmap="gray")
        plt.title("Relative threshold binary mask")
        plt.axis("off")
        plt.colorbar()
        plt.show()
        return binary_mask

    def otsu_mask(self, normalized=False):
        from skimage import filters

        if normalized:
            img = self.detector_normalised
        else:
            img = self.detector_smooth
        thresholds = filters.threshold_multiotsu(img, classes=2)
        regions = np.digitize(img, bins=thresholds)
        self.binary_mask = regions == 1
        return self.binary_mask

    def watershed_mask(self):
        markers = np.zeros_like(self.detector_smooth)
        threshold_value = np.percentile(self.detector_smooth, 80)
        markers[self.detector_smooth < threshold_value] = 1
        markers[self.detector_smooth > threshold_value] = 2
        self.binary_mask = segmentation.watershed(self.detector_smooth, markers)
        return self.binary_mask

    def edge_mask(self):
        edges = canny(self.detector_smooth, sigma=3)
        plt.imshow(edges, cmap="gray")
        fill_detector = ndi.binary_fill_holes(edges)
        self.binary_mask = ndi.binary_erosion(fill_detector, iterations=1)
        return self.binary_mask

    def plot_mask(self):
        self.otsu_mask()
        center_x, center_y = self.find_center()
        plt.imshow(self.binary_mask, cmap="gray")
        plt.plot(center_x, center_y, "r+")
        plt.title("Otsu binary mask")
        plt.axis("off")
        # plt.colorbar()

    def normalize(self, normalized=False):
        if normalized:
            img = self.detector_normalised
        else:
            img = self.detector_smooth
        binary_mask = self.otsu_mask(normalized=normalized)
        # Calculate the mean value inside and outside of the mask
        detector_value = np.mean(img[binary_mask])
        background = np.mean(img[~binary_mask])
        self.detector_normalised = (img - background) / (detector_value - background)
        self.detector_normalised[self.detector_normalised < 0] = 0
        print(
            "Detector normalised to [0,1] saved in self.detector_normalised for the Dector object,\n the orginal detector average value is: ",
            detector_value,
            " and the background average value is: ",
            background,
        )
        return detector_value, background

    def find_center(self):
        binary_mask = self.otsu_mask()
        # Find the center of the donut-shaped mask using image moments
        M = moments((binary_mask * 255).astype(np.uint8))
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y

    def radial_average(self, normalize=False):
        """Calculate the radial profile of an image from a given center."""
        if normalize:
            self.normalize(normalized=False)
            img = self.detector_normalised
        else:
            img = self.detector_smooth
        y, x = np.indices((img.shape))
        center_x, center_y = self.find_center()
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(np.int64)
        tbin = np.bincount(r.ravel(), img.ravel())
        nr = np.bincount(r.ravel())
        profile = tbin / nr
        # profile[profile<0] = 0
        return profile  # The last value is not relevant

    def plot_radial_average(self, normalize=False):
        profile = self.radial_average(normalize=normalize)
        radii = np.arange(len(profile))
        plt.plot(radii, profile, label="Radial Average")
        # plot the detector_value and background from the normalised detector
        detector_value, background = self.normalize(normalized=normalize)
        plt.axhline(y=detector_value, color="r", linestyle="--", label="Detector")
        plt.axhline(y=background, color="b", linestyle="--", label="Background")
        plt.legend()
        plt.title("Radial Averaged Intensity")
        plt.xlabel("Radius (pixels)")
        plt.ylabel("Averaged Intensity")
        plt.grid(True)

    def plot(self):
        plt.subplot(2, 2, 1)
        self.plot_orginal()
        plt.subplot(2, 2, 2)
        plt.imshow(self.detector_smooth, cmap="gray")
        plt.title("Smoothed Image")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        self.plot_mask()
        plt.subplot(2, 2, 4)
        self.plot_radial_average()
        plt.tight_layout()
        plt.show()


class Calibrate_Dose(object):
    def __init__(
        self, aperture_experiment_beam_file=None, aperture_detector_beam_file=None
    ):

        data, dimensions, calibration, metadata = dm_load(aperture_experiment_beam_file)
        self.aperature_experiment_beam = Detector(data.astype(float))

        data, dimensions, calibration, metadata = dm_load(aperture_detector_beam_file)
        self.aperature_detector_beam = Detector(data.astype(float))

        self.background = 0
        self.experiment_scan_value = 0
        self.detector_scan_value = 0
        self.scale = 1
        self.cleaned = False

    def remove_artifacts(self):
        assert (
            self.aperature_experiment_beam is not None
        ), "Experiment beam aperture not set"
        assert (
            self.aperature_detector_beam is not None
        ), "Detector beam aperture not set"
        self.aperature_detector_beam.remove_hot_pixels(sigma=30, update=True)
        self.aperature_experiment_beam.remove_hot_pixels(sigma=30, update=True)
        self.cleaned = True

    def dose_scale(self):
        self.remove_artifacts()
        print("Normalizing beam used for detector scan...")
        background_detector, detector_scan_value = (
            self.aperature_detector_beam.normalize()
        )
        print("Normalizing beam used for experiment scan...")
        background, experiment_scan_value = self.aperature_experiment_beam.normalize()
        self.background = background
        self.experiment_scan_value = experiment_scan_value
        self.detector_scan_value = detector_scan_value
        scale = (experiment_scan_value - background) / (
            detector_scan_value - background_detector
        )
        self.scale = scale
        print(f"Dose scale between experiment scan and detector scale: {scale}")
        return scale

    def plot(self):
        if not self.cleaned:
            self.remove_artifacts()
        self.aperature_detector_beam.plot()
        self.aperature_experiment_beam.plot()


class Calibrate_Detector(object):
    def __init__(self):
        self.detector = None
        self.dose_scale = None
        self.radial_average = None
        self.gain = None
        self.background = None
        self.detector_max = None

    def set_detector(self, array):
        self.detector = Detector(array)

    def read_dose_scale(
        self, aperture_experiment_beam_file=None, aperture_detector_beam_file=None
    ):
        cali_dose = Calibrate_Dose(
            aperture_experiment_beam_file, aperture_detector_beam_file
        )
        self.dose_scale = cali_dose.dose_scale()
        cali_dose.plot()

    def calibrate(self):
        assert self.detector is not None, "Detector not set"
        assert self.dose_scale is not None, "Dose scale not set"
        detector_value, background = self.detector.normalize()
        self.detector.plot()
        gain = (detector_value - background) * self.dose_scale
        detector_max = gain + background
        self.gain = gain
        self.background = background
        self.detector_max = detector_max
        print(
            f"Detector gain: {gain}, background: {background}, detector max: {detector_max}"
        )
        return gain, background, detector_max
