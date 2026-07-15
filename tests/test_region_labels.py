"""Regression tests for the region-column-label API on Fitter.

These pin down that ``region_column_labels`` is a *pure* property (no hidden
mutation of ``coordinates``/``atom_types``) and that out-of-bounds columns are
labelled ``-1`` and dropped only by the explicit prune method.
"""
import numpy as np

from qem.fit.fitter import Fitter


def _make_fitter():
    image = np.zeros((20, 20), dtype=np.float32)
    model = Fitter(image, dx=1.0, elements=["Au"])
    # Two in-bounds columns and one out-of-bounds (x = 99).
    model.coordinates = np.array([[5.0, 5.0], [10.0, 12.0], [99.0, 3.0]])
    model.atom_types = np.array([0, 0, 0])
    return model


def test_region_column_labels_is_pure():
    """Accessing the property must not mutate coordinates/atom_types."""
    model = _make_fitter()
    before_coords = model.coordinates.copy()
    before_types = model.atom_types.copy()

    labels = model.region_column_labels

    np.testing.assert_array_equal(model.coordinates, before_coords)
    np.testing.assert_array_equal(model.atom_types, before_types)
    # Aligned 1:1 with coordinates.
    assert len(labels) == len(before_coords)


def test_out_of_bounds_columns_labelled_minus_one():
    model = _make_fitter()
    labels = model.region_column_labels
    # In-bounds columns fall in the default region 0; OOB column is -1.
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == -1
    # Region masks naturally exclude out-of-bounds columns.
    assert (labels == 0).sum() == 2


def test_prune_out_of_bounds_columns():
    model = _make_fitter()
    mask = model.prune_out_of_bounds_columns()
    np.testing.assert_array_equal(mask, [True, True, False])
    assert len(model.coordinates) == 2
    assert len(model.atom_types) == 2
