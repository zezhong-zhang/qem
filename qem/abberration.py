import matplotlib.pyplot as plt
import numpy as np


def aberration_function(aberrations, wavelength, kx, ky):
    """
    Calculate the aberration phase shift for given aberrations and spatial frequencies.

    Parameters:
    aberrations (dict): Dictionary of aberration coefficients (e.g. {'A1': 1e-9, 'C3': 1e-6})
    kx, ky (ndarray): Spatial frequency coordinates in the x and y directions

    Returns:
    phase_shift (ndarray): Aberration phase shift over the spatial frequency plane
    """
    # Convert to polar coordinates
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)

    # Initialize the phase shift to zero
    phase_shift = np.zeros_like(k)

    # Aberration contributions (based on common names like A1, C3, etc.)
    for abbr, coeff in aberrations.items():
        if abbr == 'A1':  # Axial coma
            phase_shift += coeff * k * np.cos(theta)
        elif abbr == 'A2':  # Astigmatism
            phase_shift += coeff * k * np.sin(2 * theta)
        elif abbr == 'B2':  # Two-fold astigmatism
            phase_shift += coeff * k * np.cos(2 * theta)
        elif abbr == 'C3':  # Spherical aberration
            phase_shift += coeff * k**3
        elif abbr == 'C5':  # Higher order spherical aberration
            phase_shift += coeff * k**5
        elif abbr == 'A3':  # Trefoil aberration
            phase_shift += coeff * k**2 * np.sin(3 * theta)
        # Add more aberrations as needed following the convention (A1, C3, C5, etc.)

    return phase_shift

# Define constants


def contrast_transfer_function(u, defocus, Cs, wavelength):
    """
    Calculate the Contrast Transfer Function (CTF) for given parameters.

    Parameters:
    u         : array-like, spatial frequency
    defocus   : defocus value (Delta f)
    Cs        : spherical aberration coefficient
    wavelength: wavelength of the electron

    Returns:
    CTF value for the given spatial frequencies.
    """
    u = np.array(u) * 1e10  # convert to 1/m
    phase_shift = np.pi * wavelength * u**2 * (-defocus + 0.5 * Cs * wavelength**2 * u**2)
    ctf = np.sin(phase_shift)
    return ctf

# Parameters (example values)


defocus = 1e-8  # in meters
Cs = 1e-3  # spherical aberration in meters
voltage = 1.97e-12  # electron wavelength for 300 keV in meters

# Spatial frequency range
u = np.linspace(0, 1, 10000)  # in 1/A

# Calculate CTF
ctf_values = contrast_transfer_function(u, defocus, Cs, voltage)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(u, ctf_values)
plt.title('Contrast Transfer Function')
plt.xlabel('Spatial Frequency (1/A)')
plt.ylabel('CTF')
plt.grid(True)
plt.show()
