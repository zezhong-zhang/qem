"""Run qem.fusion on the vendor multi_modal tutorial demo data."""

import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem.fusion import JointLeastSquaresRoute, MultiModalDataset, save_fusion_result


DEMO_H5 = "/Users/zhangzz/code/qem/vendors/multi_modal/mapfusion/example_data/demo_EDX_maps.h5"
ELEMENTS = ["Co", "S", "O"]


def load_demo_dataset(map_index=7, stride=4):
    with h5py.File(DEMO_H5, "r") as file:
        group = file[f"map{map_index}"]
        adf = group["HAADF"][::stride, ::stride]
        maps = {element: group[element][::stride, ::stride] for element in ELEMENTS}
    return MultiModalDataset.from_chemical_maps(adf, maps, elements=ELEMENTS)


def main():
    dataset = load_demo_dataset()
    route = JointLeastSquaresRoute(
        elements=ELEMENTS,
        gamma=1.6,
        lambda_adf=1.0,
        lambda_edx=0.1,
        lambda_tv=0.0,
        step_size=0.5,
        max_iter=40,
        tolerance=0.0,
    )
    result = route.fit(dataset).get_results()
    save_fusion_result(result, "multi_modal_vendor_demo_result.npz")
    print("initial_cost", result.cost_history["total"][0])
    print("final_cost", result.cost_history["total"][-1])
    print("iterations", result.metadata["iterations"])
    return result


if __name__ == "__main__":
    main()
