"""Example ADF-EDX-EELS joint quantification pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem.fusion import MultiModalAnalyzer, save_fusion_result


DATA_DIR = "~/work/data/High_entropy/script"
ELEMENTS = ["Ti", "Fe", "Cr", "Ni", "La", "Mn", "Co", "Cu", "O"]


def main():
    analyzer = MultiModalAnalyzer.from_hspy(
        DATA_DIR,
        elements=ELEMENTS,
        eels_high_loss_file="eels_hl_aligned_bin.hspy",
    )
    result = analyzer.fit(
        gamma=1.6,
        lambda_adf=1.0,
        lambda_edx=0.2,
        lambda_eels=0.2,
        lambda_tv=0.01,
        step_size=0.02,
        max_iter=50,
    )
    save_fusion_result(result, "adf_edx_eels_fusion_result.npz")
    return result


if __name__ == "__main__":
    main()
