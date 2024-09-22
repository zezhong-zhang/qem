import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np


def add_poisson_noise(image):
    """
    Adds Poisson noise to an image.

    Parameters:
    - image: 2D array of pixel values (representing intensities).
    - key: JAX random key for reproducibility.

    Returns:
    - noisy_image: The image with Poisson noise applied.
    """
    key = random.PRNGKey(0)
    noisy_image = random.poisson(key, image)
    return noisy_image


def compute_fim(model_func, params):
    """
    Compute the Fisher Information Matrix (FIM) for a given model function with Poisson noise.

    Parameters:
    - model_func: The model function, e.g., Gaussian, Lorentzian, etc.
    - params: Parameters of the model function as a JAX array.
    - obs: Observed image data.
    - x, y: Grid of points over the 2D image.
    - dose: Dose parameter that scales the intensity (affects Poisson noise).

    Returns:
    - FIM: Fisher Information Matrix.
    """
    # Get the number of parameters
    num_params = len(params)
    # Initialize the Fisher Information Matrix (FIM)
    FIM = np.zeros((num_params, num_params))

    grads_list = []
    model_func_vals = model_func(params)
    # Compute the Fisher Information Matrix
    for i in range(num_params):
        # small perturb of the parameter
        delta = 1e-6
        params_perturb = params.at[i].set(params[i] + delta)
        grad = (model_func(params_perturb) - model_func_vals) / delta
        grads_list.append(grad)
    grads = np.array(grads_list)

    for i in range(num_params):
        for j in range(num_params):
            result = np.sum(grads[i, :, :] * grads[j, :, :] / model_func_vals)
            FIM[i, j] = result
    return FIM


def compute_crb(fim):
    """
    Compute the Cramer-Rao Bound (CRB) given a Fisher Information Matrix (FIM).

    Parameters:
    - fim: Fisher Information Matrix.

    Returns:
    - crb: Cramer-Rao Bound for each parameter.
    """
    FIM_inv = np.linalg.pinv(fim)
    vars = np.diag(FIM_inv)
    crb = np.sqrt(vars)
    return crb


def joint_probability_2d(observations, params, model_func):
    """
    Compute the joint probability P(omega | theta) for a 2D image with Poisson-distributed data.

    Parameters:
    - observations: The observed 2D image data (e.g., pixel values) as a JAX array.
    - params: Parameters of the model function as a JAX array.
    - model_func: The model function to compute lambda_k.
    - x, y: Grid of points over the 2D image (same shape as observations).
    - dose: Dose parameter that scales the intensity (affects Poisson noise).

    Returns:
    - Joint probability P(omega | theta) for the entire 2D image.
    """
    # Compute the expected values lambda_k (same shape as the observations)
    lambda_k = model_func(params)

    # Compute the individual probabilities for each pixel
    individual_probs = (
        (lambda_k**observations)
        * jnp.exp(-lambda_k)
        / jax.scipy.special.factorial(observations)
    )

    # Compute the joint probability by taking the product of all pixel probabilities
    joint_prob = jnp.prod(individual_probs)

    return joint_prob
