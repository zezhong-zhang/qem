import h5py

from qem.fusion import JointLeastSquaresRoute, MultiModalDataset


DEMO_H5 = "/Users/zhangzz/code/qem/vendors/multi_modal/mapfusion/example_data/demo_EDX_maps.h5"


def test_vendor_multi_modal_demo_data_cost_decreases():
    elements = ["Co", "S", "O"]
    with h5py.File(DEMO_H5, "r") as file:
        group = file["map7"]
        adf = group["HAADF"][::8, ::8]
        maps = {element: group[element][::8, ::8] for element in elements}

    dataset = MultiModalDataset.from_chemical_maps(adf, maps, elements=elements)
    route = JointLeastSquaresRoute(
        elements=elements,
        lambda_adf=1.0,
        lambda_edx=0.1,
        lambda_tv=0.0,
        step_size=0.5,
        max_iter=20,
        tolerance=0.0,
    )
    result = route.fit(dataset).get_results()

    assert result.concentrations.shape == (64, 64, 3)
    assert result.cost_history["total"][-1] < result.cost_history["total"][0]
