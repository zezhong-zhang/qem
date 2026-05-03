import numpy as np

from qem.fusion import (
    JointLeastSquaresRoute,
    MultiModalAnalyzer,
    MultiModalDataset,
    load_fusion_result,
    save_fusion_result,
)


def _synthetic_dataset():
    yy, xx = np.mgrid[:8, :9]
    c0 = 0.25 + 0.5 * (xx / xx.max())
    c1 = 0.2 + 0.4 * (yy / yy.max())
    concentrations = np.stack([c0, c1], axis=-1)
    edx_ref = np.array(
        [
            [1.0, 0.05],
            [0.2, 0.7],
            [0.0, 0.4],
        ]
    )
    eels_ref = np.array(
        [
            [0.8, 0.1],
            [0.1, 0.9],
        ]
    )
    return MultiModalDataset.synthetic(
        concentrations,
        elements=["Ti", "Ni"],
        edx_reference=edx_ref,
        eels_reference=eels_ref,
        adf_weights=[22.0, 28.0],
        gamma=1.6,
    ), concentrations


def test_joint_route_reduces_synthetic_cost():
    dataset, _ = _synthetic_dataset()
    route = JointLeastSquaresRoute(
        elements=dataset.elements,
        adf_weights=[22.0, 28.0],
        lambda_adf=0.1,
        lambda_edx=1.0,
        lambda_eels=1.0,
        step_size=0.1,
        max_iter=40,
    )
    route.fit(dataset)
    result = route.get_results()

    assert result.concentrations.shape == (8, 9, 2)
    assert np.all(result.concentrations >= 0)
    assert result.cost_history["total"][-1] <= result.cost_history["total"][0]
    assert set(result.as_maps()) == {"Ti", "Ni"}


def test_multimodal_analyzer_entry_point():
    dataset, _ = _synthetic_dataset()
    analyzer = MultiModalAnalyzer(dataset)
    result = analyzer.fit(
        adf_weights=[22.0, 28.0],
        lambda_adf=0.1,
        lambda_edx=1.0,
        lambda_eels=1.0,
        step_size=0.1,
        max_iter=5,
    )
    assert result.metadata["iterations"] >= 1


def test_fusion_result_round_trip(tmp_path):
    dataset, _ = _synthetic_dataset()
    result = JointLeastSquaresRoute(
        elements=dataset.elements,
        adf_weights=[22.0, 28.0],
        max_iter=2,
    ).fit(dataset).get_results()

    path = tmp_path / "fusion_result.npz"
    save_fusion_result(result, path)
    loaded = load_fusion_result(path)

    np.testing.assert_allclose(loaded.concentrations, result.concentrations)
    assert loaded.elements == result.elements
    assert loaded.metadata["gamma"] == result.metadata["gamma"]
