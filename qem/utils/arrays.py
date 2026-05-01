"""
Array manipulation utilities for QEM.
"""

import numpy as np


def find_duplicate_row_indices(array):
    """
    Find the indices of duplicate rows in a NumPy array.

    Parameters:
    - array: NumPy array to check for duplicate rows.

    Returns:
    - idx_duplicates: NumPy array of indices of the duplicate rows.
    """
    # Step 1: Find indices of unique rows
    _, idx_unique = np.unique(array, axis=0, return_index=True)

    # Step 2: Initialize a mask assuming all are duplicates
    mask = np.ones(len(array), dtype=bool)

    # Step 3: Mark unique rows as not duplicates
    mask[idx_unique] = False

    # Step 4: Find indices of duplicates
    idx_duplicates = np.where(mask)[0]

    return idx_duplicates


def find_row_indices(array1, array2):
    """
    Efficiently find the indices of rows of array1 in array2.

    :param array1: A NumPy array of shape (M, N).
    :param array2: A NumPy array of shape (K, N).
    :return: A NumPy array of indices indicating the position of each row of array1 in array2.
             If a row from array1 is not found in array2, the corresponding index is -1.
    """
    # Create a dictionary mapping from row tuple to index for array2
    row_to_index = {tuple(row): i for i, row in enumerate(array2)}

    # Find indices for rows in array1 using the dictionary
    indices = np.array([row_to_index.get(tuple(row), -1) for row in array1])

    return indices


def find_element_indices(array1, array2):
    """
    Find indices of elements of array1 in array2.

    :param array1: A 1D NumPy array of elements to find.
    :param array2: A 1D NumPy array where to search for the elements.
    :return: A list of indices indicating the position of each element of array1 in array2.
             If an element from array1 is not found in array2, the corresponding index is -1.
    """
    indices = [
        (
            np.where((array2 == element).all(axis=1))[0][0]
            if (array2 == element).all(axis=1).any()
            else -1
        )
        for element in array1
    ]
    indices = np.array(indices)
    # drop the -1s
    indices = indices[indices != -1]
    return indices


def get_random_indices_in_batches(total_examples, batch_size):
    """Get random indices split into batches."""
    all_indices = np.random.permutation(total_examples)
    batches = [
        all_indices[i: i + batch_size] for i in range(0, total_examples, batch_size)
    ]
    return batches