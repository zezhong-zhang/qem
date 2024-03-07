#%%
import sys
sys.path
sys.path.append('/media/chu-ping/project/ from_linux/pyStatSTEM')

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
I.fit_gaussian(idiv_sig=True, extend=0)
I.plot_image()

# %%
volume = I.gaussian_prmt[0]['weight']*I.gaussian_prmt[0]['sig']
coor = I.gaussian_prmt[0]['mean']
# &&
g = ps.GaussianMixtureModelObject(volume, coor)
g.remove_edge(20, [I.nx, I.ny])
g.GMM([1, 40], criteria=['nllh', 'icl', 'clc', 'bic'], pos_init=(1,0))
g.plot_criteria(['icl', 'clc', 'bic'])
# %%
uc = 35
g.plot_histogram(use_component=uc)
g.plot_thickness(use_component=uc)
# %%
