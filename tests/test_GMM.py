#%%
import sys
sys.path.insert(0,'D:/project/ from_linux/pyStatSTEM')
import numpy as np
import matplotlib.pyplot as plt
import qem
import pytest

def test_GMM(path_to_data="./Examples/Example_PtIr.mat", plot=False):
    # Load data for comparison
    legacyStatSTEM = qem.io.read_legacyInputStatSTEM(path_to_data)
    outputStatSTEM = legacyStatSTEM["output"]
    atomcountsStatSTEM = legacyStatSTEM["atomcounting"]

    columns = qem.AtomicColumns(
        "HAADF", outputStatSTEM["volumes"], outputStatSTEM["coordinates"][:, 0:2]
    )

    # For compatibility with GMM, but we should probably adapt these conventions
    # We could include the GMM into AtomicColumns class as well or make GMM a subclass of it
    # columns.scs["HAADF"] = np.transpose(np.expand_dims(columns.scs["HAADF"], -1))
    columns.scs["HAADF"] = np.expand_dims(columns.scs["HAADF"], -1)
    columns.pos = np.transpose(columns.pos)

    # gmm_obj = pyStatSTEM.GaussianMixtureModelObject(columns.scs["HAADF"], columns.pos)
    # gmm_obj.GMM(20, criteria=["icl"], pos_init=(1, 0))
    gmm_obj = qem.GaussianMixtureModel(columns.scs["HAADF"])
    gmm_obj.GMM("test", 20, score_method=['icl'])
    
    if plot:
        # Plot the results for comparison (should be deleted later)
        plt.plot(range(1, 21), atomcountsStatSTEM["ICL"], "-k")
        # plt.plot(range(1, 21), gmm_obj.criteria_dict["icl"], "or")
        plt.plot(range(1, 21), gmm_obj.result["test"].score["icl"], "or")
        plt.legend(["ICL-Matlab version", "ICL-Python version"])
        plt.xticks(range(1, 21))
        plt.grid()
        plt.show()

    # Assert that the results are roughly the same
    # Thats the essence of automated pytest testing
    # It should make sure the function is working as expected
    # and be compared to the results from the legacy code
    # or known results
    assert np.all(
        gmm_obj.result["test"].score["icl"] == pytest.approx(atomcountsStatSTEM["ICL"], rel=1e-2)
    )

# %%
test_GMM(plot=True)
# %%
