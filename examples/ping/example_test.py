# %%
import sys

# sys.path
# sys.path.append('D:\project\ from_linux\pyStatSTEM')
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import qem.classes as ps
import qem
import timeit

# matplotlib.use('Tkagg')

# %%
# READ DATA FROM FILE
# requried variable: raw image, fitted model (for comparison)

# path_to_data="/home/chu-ping/Documents/projects/multivariance_model/samples/AuAg/au_rod_7_adf1_result_samew.mat"
path_to_data = "/home/zzhang/OneDrive/code/qem/examples/Example_Au.mat"
legacyStatSTEM = qem.io.read_legacyInputStatSTEM(path_to_data)
inputStatSTEM = legacyStatSTEM["input"]
outputStatSTEM = legacyStatSTEM["output"]
atomcountsStatSTEM = legacyStatSTEM["atomcounting"]
image = inputStatSTEM["obs"]
model = outputStatSTEM["model"]
coor = inputStatSTEM["coordinates"].T[0:2, :]
dx = outputStatSTEM["dx"]
coor = np.flip(coor, 0) / dx
icl = atomcountsStatSTEM["ICL"]
nllh = atomcountsStatSTEM["mLogLik"]

# %%
starttime = timeit.default_timer()
I1_same = ps.ImageProcess(image, coor=coor)
I1_same.fit_gaussian()
I1_same.plot_image()
print("Time :", timeit.default_timer() - starttime)
# %%
starttime = timeit.default_timer()
I1_change = ps.ImageProcess(image, coor=coor)
I1_change.fit_gaussian()
I1_change.plot_image()
print("Time :", timeit.default_timer() - starttime)
# %%
# COMPARE WITH PREVIOUS RESULT

fig, ax = plt.subplots(1, 3)
ax[0].imshow(model - image)
ax[1].imshow(I1_same.image_fitted[0] - I1_same.image[0])
ax[2].imshow(I1_change.image_fitted[0] - I1_change.image[0])
print(
    "MSE original: ",
    np.array((model - image) ** 2).mean(),
    "\n",
    "MSE python1: ",
    np.array((I1_same.image_fitted[0] - I1_same.image[0]) ** 2).mean(),
    "\n",
    "MSE python2: ",
    np.array((I1_change.image_fitted[0] - I1_change.image[0]) ** 2).mean(),
    "\n",
    "crystal only \n",
    "MSE original: ",
    np.array(
        (
            model[I1_same.use_coor[0], I1_same.use_coor[1]]
            - image[I1_same.use_coor[0], I1_same.use_coor[1]]
        )
        ** 2
    ).mean(),
    "\n",
    "MSE python1: ",
    np.array(
        (
            I1_same.image_fitted[0][I1_same.use_coor[0], I1_same.use_coor[1]]
            - I1_same.image[0][I1_same.use_coor[0], I1_same.use_coor[1]]
        )
        ** 2
    ).mean(),
    "\n",
    "MSE python2: ",
    np.array(
        (
            I1_change.image_fitted[0][I1_same.use_coor[0], I1_same.use_coor[1]]
            - I1_change.image[0][I1_same.use_coor[0], I1_same.use_coor[1]]
        )
        ** 2
    ).mean(),
    "\n",
)
fig.dpi = 300

# %%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(model[100:200, 400:600])
ax[1].imshow(I1_same.image_fitted[0][100:200, 400:600], vmin=105.4548570448642)
ax[2].imshow(image[100:200, 400:600])
# ax[2].imshow(model[100:200,400:600]-I1_same.image_fitted[0][100:200,400:600])
fig.dpi = 300

# %%
from qem.gaussian_mixture_model import GaussianMixtureModelObject

starttime = timeit.default_timer()
volume = I1_same.gaussian_prmt[0]["weight"]
coor = I1_same.gaussian_prmt[0]["mean"]
g1_same = GaussianMixtureModelObject(volume, coor)
g1_same.remove_edge(50, [I1_same.nx, I1_same.ny])
g1_same.GMM(max_component=40, criteria=["nllh", "icl", "clc", "bic"], pos_init=(1, 0))
print("Time :", timeit.default_timer() - starttime)
# %%
starttime = timeit.default_timer()
volume = I1_change.gaussian_prmt[0]["weight"] * (I1_change.gaussian_prmt[0]["sig"] ** 2)
coor = I1_change.gaussian_prmt[0]["mean"]
g1_change = GaussianMixtureModelObject(volume, coor)
g1_change.remove_edge(50, [I1_same.nx, I1_same.ny])
g1_change.GMM(max_component=50, criteria=["nllh", "icl", "clc", "bic"], pos_init=(1, 0))
print("Time :", timeit.default_timer() - starttime)
# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.arange(1, 51), icl)
g1_same.plot_criteria(["icl"])
# g1_change.plot_criteria(['icl'])
# %%
plt.figure()
plt.plot(np.arange(1, 51), nllh)
g1_same.plot_criteria(["nllh"])
g1_change.plot_criteria(["nllh"])


# %%
path_to_data = "/home/chu-ping/Documents/projects/multivariance_model/samples/AuDopeAg/Au_dope_Ag_img.h5"

f = h5py.File(path_to_data)
img_adf1 = f["adf1"]
img_adf2 = f["adf2"]

# %%
I1 = ps.ImageProcess(img_adf1)
I1.find_peak(th_dist=6, th_inten=0.05, b_user_confirm=True)
I1.fit_gaussian(idiv_sig=True, extend=0)
I1.plot_image()

I2 = ps.ImageProcess(img_adf2, coor=I1.coordinate[0])
I2.fit_gaussian(idiv_sig=True)
I2.plot_image()

# %%
volume = I1.gaussian_prmt[0]["weight"] * (I1.gaussian_prmt[0]["sig"] ** 2)
coor = I1.gaussian_prmt[0]["mean"]
g1 = ps.GaussianMixtureModelObject(volume, coor)
g1.remove_edge(10, [I1.nx, I1.ny])
g1.GMM([1, 20], criteria=["nllh", "icl", "clc", "bic"], pos_init=(1, 0))

volume = I2.gaussian_prmt[0]["weight"] * (I2.gaussian_prmt[0]["sig"] ** 2)
coor = I2.gaussian_prmt[0]["mean"]
g2 = ps.GaussianMixtureModelObject(volume, coor)
g2.remove_edge(10, [I2.nx, I2.ny])
g2.GMM([1, 20], criteria=["nllh", "icl", "clc", "bic"], pos_init=(1, 0))

volume = np.vstack(
    [
        I1.gaussian_prmt[0]["weight"] * (I1.gaussian_prmt[0]["sig"] ** 2),
        I2.gaussian_prmt[0]["weight"] * (I2.gaussian_prmt[0]["sig"] ** 2),
    ]
)
coor = I2.gaussian_prmt[0]["mean"]
g12 = ps.GaussianMixtureModelObject(volume, coor)
g12.remove_edge(10, [I2.nx, I2.ny])
g12.GMM([1, 20], criteria=["nllh", "icl", "clc", "bic"], pos_init=(1, 0))
# %%
g1.plot_criteria(["icl"])
g2.plot_criteria(["icl"])
g12.plot_criteria(["icl"])
# %%
uc = 6

g1.plot_thickness(uc)
g2.plot_thickness(uc)
g12.plot_thickness(uc)
# %%
plt.figure()
plt.scatter(
    g1.coordinates[0, :], g1.coordinates[1, :], abs(g1.thickness - g2.thickness)
)
# %%
g12.plot_histogram(16)
# %%
g12.plot_thickness(16, show_component=[5, 8, 9, 10, 11, 12, 13, 16])
# %%
hf = h5py.File("au_rod_7_py.h5", "w")
h5data = hf.create_dataset("adf1_w", I1_change.gaussian_prmt[0]["weight"].shape)
h5data[:] = I1_change.gaussian_prmt[0]["weight"]
h5data = hf.create_dataset("adf1_s", I1_change.gaussian_prmt[0]["sig"].shape)
h5data[:] = I1_change.gaussian_prmt[0]["sig"]
h5data = hf.create_dataset("adf1_m", I1_change.gaussian_prmt[0]["mean"].shape)
h5data[:] = I1_change.gaussian_prmt[0]["mean"]
h5data = hf.create_dataset("adf1_img", I1_change.image[0].shape)
h5data[:] = I1_change.image[0]
hf.close()
# %%
