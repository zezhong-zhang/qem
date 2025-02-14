import numba as nb
import numpy as np
from hyperspy.signals import BaseSignal, Signal2D
from skimage.segmentation import watershed
from tqdm import tqdm as progressbar


def voronoi_integrate(
    s,
    points_x,
    points_y,
    method="Voronoi",
    max_radius="Auto",
    show_progressbar=True,
    remove_edge_cells=False,
    edge_pixels=1,
    pbc=False,
):
    """Given a spectrum image a set of points and a maximum outer radius,
    this function integrates around each point in an image, using either
    Voronoi cell or watershed segmentation methods.

    Parameters
    ----------
    s : HyperSpy signal or array-like object
        Assuming 2D, 3D or 4D dataset where the spatial dimensions are 2D and
        any remaining dimensions are spectral.
    point_x, point_y : list
        Detailed list of the x and y coordinates of each point of
        interest within the image.
    method : string
        'Voronoi' or 'Watershed'
    max_radius : {'Auto'} int
        A maximum outer radius for each Voronoi Cell.
        If a pixel exceeds this radius it will not be included in the cell.
        This allows analysis of a surface and particles.
        If 'max_radius' is left as 'Auto' then it will be set to the largest
        dimension in the image.
    remove_edge_cells : bool
        Determine whether to replace the cells touching the signal edge with
        np.nan values, which makes automatic contrast estimation easier later
    edge_pixels : int
        Only used if remove_edge_cells is True. Determines the number of
        pixels from the border to remove.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    integrated_intensity : NumPy array
        An array where dimension 0 is the same length as points, and subsequent
        subsequent dimension are energy dimensions.
    intensity_record : HyperSpy signal, same size as s
        Each pixel/voxel in a particular segment or region has the value of the
        integration, value.
    point_record : HyperSpy Signal2D with no navigation dimension
        Image showing where each integration region is, pixels equating to
        point 0 (integrated_intensity[0]) all have value 0, all pixels
        equating to integrated_intensity[1] all have value 1 etc.

    Examples
    --------

    >>> import atomap.api as am
    >>> from atomap.tools import integrate
    >>> import hyperspy.api as hs
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice(
    ...        image_noise=True)
    >>> image = hs.signals.Signal2D(sublattice.image)
    >>> i_points, i_record, p_record = integrate(
    ...        image,
    ...        points_x=sublattice.x_position,
    ...        points_y=sublattice.y_position, method='Voronoi')
    >>> i_record.plot()

    For a 3 dimensional dataset, with artificial EELS data

    >>> s = am.dummy_data.get_eels_spectrum_survey_image()
    >>> s_eels = am.dummy_data.get_eels_spectrum_map()
    >>> peaks = am.get_atom_positions(s, separation=4)
    >>> i_points, i_record, p_record = integrate(
    ...         s_eels, peaks[:, 0], peaks[:, 1], max_radius=3)

    Note
    ----
    Works in principle with 3D and 4D data sets but will quickly hit a
    memory error with large sizes.

    """
    image = s.__array__()
    if len(image.shape) < 2:
        raise ValueError("s must have at least 2 dimensions")
    intensity_record = np.zeros_like(image, dtype=float)
    integrated_intensity = np.zeros(image.shape[:-2])
    integrated_intensity = np.stack(
        [integrated_intensity for i in range(len(points_x))]
    )

    points = np.array((points_y, points_x))
    # Setting max_radius to the width of the image, if none is set.
    if method == "Voronoi":
        if max_radius == "Auto":
            max_radius = max(image.shape[-2:])
        elif max_radius <= 0:
            raise ValueError("max_radius must be higher than 0.")
        point_record = calculate_point_record(image, points, max_radius, pbc=pbc)

    elif method == "Watershed":
        if len(image.shape) > 2:
            raise ValueError(
                "Currently Watershed method is only implemented for 2D data."
            )
        points_map = _make_mask(image.T, points[0], points[1])
        point_record = watershed(-image, points_map.T)

    else:
        raise NotImplementedError("Oops! You have asked for an unimplemented method.")
    point_record -= 1
    for point_index in progressbar(
        range(points.shape[1]), desc="Integrating", disable=not show_progressbar
    ):
        integrated_intensity[point_index] = get_integrated_intensity(
            point_record, image, point_index
        )

    for i, j in progressbar(
        np.ndindex(image.shape[-2:]),
        desc="Building intensity map",
        total=np.prod(image.shape[-2:]),
        leave=False,
    ):

        point_index = point_record[i, j]
        if point_index == -1:
            intensity_record[..., i, j] = np.nan
        else:
            summed = integrated_intensity[point_index]
            intensity_record[..., i, j] = summed

    if isinstance(s, BaseSignal):
        intensity_record = s._deepcopy_with_new_data(
            intensity_record, copy_variance=True
        )
    else:
        intensity_record = Signal2D(intensity_record)

    point_record = Signal2D(point_record)
    point_record.metadata.Signal.quantity = "Column Index"
    intensity_record.metadata.Signal.quantity = "Integrated Intensity"

    if remove_edge_cells:
        remove_integrated_edge_cells(
            integrated_intensity,
            intensity_record,
            point_record,
            edge_pixels=edge_pixels,
            use_nans=True,
            inplace=True,
        )
    return integrated_intensity, intensity_record, point_record


def calculate_point_record(image, points, max_radius, pbc=False):
    """
    Creates a Voronoi array where equal values belong to
    the same Voronoi cell

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    points: Array like of shape (2,N)
    max_radius: Integer, max radius of each Voronoi Cell

    Returns
    -------
    point_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    point_record = np.zeros(image.shape[-2:], dtype=int)
    for i, j in progressbar(
        np.ndindex(point_record.shape),
        desc="Calculating Voronoi",
        total=np.prod(point_record.shape),
        leave=False,
    ):
        minIndex, distMin = find_smallest_distance(
            i, j, points, image_shape=image.shape, pbc=pbc
        )
        if distMin >= max_radius:
            point_record[i][j] = 0
        else:
            point_record[i][j] = minIndex + 1
    return point_record


def get_integrated_intensity(point_record, image, point_index, include_edge_cells=True):
    """
    Using a Voronoi point_record array, integrate a (minimum 2D)
    image array at each pixel

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    image : The ndarray to integrate the voronoi cells on
    point_index: Array like of shape (2,N)

    Returns
    -------
    integrated_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    currentMask = point_record == point_index
    currentFeature = currentMask * image
    integrated_record = np.sum(currentFeature, axis=(-1, -2))
    return integrated_record


@nb.jit()
def find_smallest_distance(i, j, points, image_shape=None, pbc=False):
    """
    Finds the smallest distance between coordinates (i, j)
    and a list of coordinates.

    Parameters
    ----------
    i : Integer
    j : Integer
    points : array like of shape (2,N)

    Returns
    -------
    distMin  : Minimum distance
    minIndex : Index of minimum distance in points

    Example
    -------
    >>> import numpy as np
    >>> points = np.random.random((2, 10000))
    >>> i, j = 0.5, 0.5
    >>> smallest_distance = find_smallest_distance(i, j, points)

    """
    if pbc:
        height, width = image_shape
        distance_log = np.inf * np.ones(points.shape[1])
        for k in range(points.shape[1]):
            dx = min(abs(points[0, k] - i), width - abs(points[0, k] - i))
            dy = min(abs(points[1, k] - j), height - abs(points[1, k] - j))
            distance_log[k] = (dx**2 + dy**2) ** 0.5
    else:
        distance_log = (
            (points[0] - float(i)) ** 2 + (points[1] - float(j)) ** 2
        ) ** 0.5
    minIndex = np.argmin(distance_log)
    distMin = distance_log[minIndex]
    return minIndex, distMin


def remove_integrated_edge_cells(
    i_points, i_record, p_record, edge_pixels=1, use_nans=True, inplace=False
):
    """Removes any cells that touch within a number of pixels of
    the image border.

    Note on using use_nans: If this is used on a dataset with more than
    two dimensions, the resulting HyperSpy i_record signal might be needed to
    be viewed with i_record.plot(navigator='slider'), since HyperSpy may throw
    an error when plotting a dataset with only NaNs present.

    Parameters
    ----------
    i_points : NumPy array
        The output of the Atomap integrate function or method
    i_record : HyperSpy signal
        The output of the Atomap integrate function or method
    p_record : HyperSpy signal
        The output of the Atomap integrate function or method

    Returns
    -------
    i_points : NumPy array
        Modified list of integrated intensities with either np.nan or 0
        on the removed values, which preserves the atom index.
    i_record : HyperSpy signal
        Modified integrated intensity record, with either np.nan or 0
        on the removed values, which preserves the atom index
    p_record : HyperSpy signal, same size as image
        Modified points record, where removed areas have value = -1.

    Example
    -------

    >>> s = am.dummy_data.get_fantasite()
    >>> points_x, points_y = am.get_atom_positions(s).T
    >>> i, ir, pr = am.integrate(
    ...    s,
    ...    points_x,
    ...    points_y,
    ...    method='Voronoi',
    ...    remove_edge_cells=False)
    >>> from atomap.tools import remove_integrated_edge_cells
    >>> i2, ir2, pr2 = remove_integrated_edge_cells(
    ...    i, ir, pr, edge_pixels=5, use_nans=True)

    """
    if not inplace:
        i_points = i_points.copy()
        i_record = i_record.deepcopy()
        p_record = p_record.deepcopy()

    border = _border_elems(p_record.data, edge_pixels)
    border_indices = np.array(list(set(border)))
    indices = np.in1d(p_record.data, border_indices)
    indices = indices.reshape(p_record.data.shape)
    i_points[border_indices] = np.nan if use_nans else 0
    i_record.data[..., indices] = np.nan if use_nans else 0
    p_record.data[indices] = -1

    if not inplace:
        return i_points, i_record, p_record


def _make_mask(image, points_x, points_y):
    """
    Create points_map for the watershed integration
    function
    """
    mask = np.zeros(image.shape[-2:])
    indices = np.round(np.array([points_y, points_x])).astype(int)
    values = np.arange(len(points_x))
    mask[tuple(indices)] = values
    return mask


def _border_elems(image, pixels=1):
    """
    Return the values of the edges along the border of the image, with
    border width `pixels`.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([
    ...     [1,1,1],
    ...     [2,5,3],
    ...     [4,4,4]])
    >>> b = _border_elems(a, pixels=1)

    """
    arr = np.ones_like(image, dtype=bool)
    arr[pixels: -1 - (pixels - 1), pixels: -1 - (pixels - 1)] = False
    return image[arr]
