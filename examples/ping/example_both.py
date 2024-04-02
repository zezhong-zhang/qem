#%%
import sys
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import qem.classes as ps
import qem
# matplotlib.use('Tkagg')


path_to_data="./Examples/ping/au_rod.h5"
path_to_data="./au_rod.h5"

f = h5py.File(path_to_data)
img_adf = f['adf']['image'][:]

# %%
I = ps.ImageProcess(img_adf)
I.find_peak(th_dist=6, th_inten=0.2, b_user_confirm=False)
I.fit_gaussian()
I.plot_image()

# %%
from qem.gaussian_mixture_model import GaussianMixtureModel
volume = I.gaussian_prmt[0]['weight']*I.gaussian_prmt[0]['sig']
coor = I.gaussian_prmt[0]['mean']
# &&
g = GaussianMixtureModel(volume.reshape(-1, 1))
g.GMM(40, use_scs_channel=[0], 
    score_method=['icl', 'clc', 'bic'], init_method='equionce',constraint=['uni_width'])
# g.plot_criteria(['icl', 'clc', 'bic'])
# %%
g.import_coordinates(coor)
uc = 24
g.plot_thickness(n_component=uc,show_component=[10,15,20])
# %%
g.plot_histogram(n_component=uc,use_dim=1,bin=100)
# %%
g.plot_criteria(['icl', 'clc', 'bic'])
# %%
g.plot_criteria(['clc'])
# %%
