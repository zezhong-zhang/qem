import numpy as np


def zoom_on_pixel(input_array, coordinates, upsample_factor=1, output_shape=None,
                  num_threads=1, use_numpy_fft=False, return_real=True,
                  return_coordinates=False):
    """
    Zoom in on a 1D or 2D array using Fourier upsampling.

    Parameters
    ----------
    input_array (np.ndarray): Input 1D or 2D array to zoom into.
    coordinates (tuple of floats): Pixel coordinates to center the zoom.
    upsampling_factor (int, optional): Upsampling factor for higher resolution. Defaults to 1.
    output_shape (tuple of ints, optional): Shape of the zoomed output array. Defaults to the shape of `input_array`.
    num_threads (int, optional): Number of threads to use for FFT calculations. Defaults to 1.
    use_numpy_fft (bool, optional): If True, use NumPy's FFT implementation instead of FFTW. Defaults to False.
    return_real (bool, optional): If True, return the real part of the zoomed array. Otherwise, return the complex array. Defaults to True.
    return_coordinates (bool, optional): If True, return the zoomed coordinates along with the zoomed array. Defaults to False.

    Returns
    -------
    zoomed_array (np.ndarray): The zoomed array upsampled by `upsampling_factor` with size `output_shape`.
    coordinates (np.ndarray, optional): The coordinates of the zoomed region, if `return_coordinates` is True.
    """
    input_shape = input_array.shape
    if output_shape is None:
        output_shape = input_shape

    coordinate_grid = np.zeros((input_array.ndim,) + tuple(output_shape), dtype='float')
    for dim, (_input_size, output_size, target) in enumerate(zip(input_shape, output_shape, coordinates)):
        zoom_range = np.linspace(
            target - (output_size - 1) / (2 * upsample_factor),
            target + (output_size - 1) / (2 * upsample_factor),
            output_size
        )
        slice_spec = [None] * dim + [slice(None)] + [None] * (input_array.ndim - 1 - dim)
        coordinate_grid[dim] = zoom_range[tuple(slice_spec)]

    if input_array.ndim == 2:
        result = fourier_interp2d(
            input_array, coordinate_grid,
            nthreads=num_threads, use_numpy_fft=use_numpy_fft, return_real=return_real
        )
    else:
        raise NotImplementedError("Zooming for dimensions > 2 is not yet supported.")

    return (coordinate_grid, result) if return_coordinates else result


def zoom_nd(input_array, offsets=(), center_convention=float, **kwargs):
    """
    Zoom in on the center of a 1D or 2D array using Fourier upsampling.

    Parameters
    ----------
    input_array (np.ndarray): Input 1D or 2D array to zoom into.
    offsets (tuple of floats, optional): Offsets from the center in original pixel units. Defaults to (0, 0).
    center_convention (function, optional): Function to determine the "center" of the array. Defaults to float.
    **kwargs: Additional arguments passed to `zoom_on_pixel`.

    Returns
    -------
    zoomed_array (np.ndarray): The zoomed array upsampled by `upsampling_factor` with size `output_shape`.
    coordinates (np.ndarray, optional): The coordinates of the zoomed region, if `return_coordinates` is True.
    """
    if len(offsets) > 0 and len(offsets) != input_array.ndim:
        raise ValueError("The number of offsets must match the dimensions of the input array.")
    elif not offsets:
        offsets = (0,) * input_array.ndim

    center_coordinates = [
        center_convention((size - 1) / 2) + offset
        for size, offset in zip(input_array.shape, offsets)
    ]

    return zoom_on_pixel(input_array, center_coordinates, **kwargs)


def fourier_interp2d(data, outinds, nthreads=1, use_numpy_fft=False,
                     return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without raising
    any exceptions) data.

    Parameters
    ----------
    data : ndarray
        The data values of the array to interpolate
    outinds : ndarray
        The coordinate axis values along which the data should be interpolated
        CAN BE: `ndim x [n,m,...]` like np.indices OR (less memory intensive,
        more processor intensive) `([n],[m],...)`
    """

    # load fft
    # fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if hasattr(outinds, 'ndim') and outinds.ndim not in (data.ndim+1, data.ndim):
        raise ValueError("Must specify an array of output indices with # of dimensions = input # of dims + 1")
    elif len(outinds) != data.ndim:
        raise ValueError("outind array must have an axis for each dimension")

    fft_data = np.fft.ifft2(data)

    freqY = np.fft.fftfreq(data.shape[0])
    if hasattr(outinds, 'ndim') and outinds.ndim == 3:
        # if outinds = np.indices(shape), we extract just lines along each index
        indsY = freqY[np.newaxis, :]*outinds[0, :, 0][:, np.newaxis]
    else:
        indsY = freqY[np.newaxis, :]*np.array(outinds[0])[:, np.newaxis]
    kerny = np.exp((-1j*2*np.pi)*indsY)

    freqX = np.fft.fftfreq(data.shape[1])
    if hasattr(outinds, 'ndim') and outinds.ndim == 3:
        # if outinds = np.indices(shape), we extract just lines along each index
        indsX = freqX[:, np.newaxis]*outinds[1, 0, :][np.newaxis, :]
    else:
        indsX = freqX[:, np.newaxis]*np.array(outinds[1])[np.newaxis, :]
    kernx = np.exp((-1j*2*np.pi)*indsX)

    result = np.dot(np.dot(kerny, fft_data), kernx)

    if return_real:
        return result.real
    else:
        return result
