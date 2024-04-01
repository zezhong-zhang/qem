#%%
import sys
# sys.path.insert(0,'D:/project/ from_linux/pyStatSTEM')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from qem.gaussian_mixture_model import GaussianMixtureModel

#%%
dir = '.'
# dir = './Examples/ping'
filename = 'au_rod.h5'
f = h5py.File(dir + '/' + filename, 'r')
coordinate = f['adf']['coordinate'][:]
volume1 = np.squeeze(f['adf']['volume'][:])
volume2 = np.squeeze(f['abf']['volume'][:])
img_adf = f['adf']['image'][:]

#%%
volume = np.array([volume1, volume2]).T
h = GaussianMixtureModel(volume)
h.fit("0+1_nomean", 40, use_scs_channel=[0,1], 
    score_method=['icl', 'clc', 'bic'], init_method='equionce')
# h.GMM("0", 40, use_scs_channel=[0], 
#     score_method=['icl', 'clc', 'bic'])
# h.GMM("1", 40, use_scs_channel=[1], 
#     score_method=['icl', 'clc', 'bic'])

#%%
name = ["0+1_nomean"]
f, ax = plt.subplots(1,3, sharey=False)
ax[0].plot(h.result[name[0]].score['icl'])
ax[1].plot(h.result[name[0]].score['nllh'])
ax[2].plot(h.result[name[0]].score['bic'])


# %%
f, ax = plt.subplots(1,1, sharey=False)
plt.scatter(h.result[name[0]].val[:,0], h.result[name[0]].val[:,1])
plt.scatter(h.result[name[0]].mean[20][:,0], h.result[name[0]].mean[20][:,1], marker="x")



#%%
# Use ADF
# g = ps.GaussianMixtureModelObject(volume1, coordinate)
# g.GMM([1, 40], criteria=['nllh', 'icl', 'clc', 'bic'], pos_init=(1,0))
# g.plot_criteria(['icl', 'clc', 'bic'])
# uc = 31
# g.plot_historgram(use_component=uc)
# g.plot_thickness(use_component=31, show_component=20)


volume = np.array([volume1]).T
h = GaussianMixtureModel(volume)
h.fit("0", 40, use_scs_channel=[0], 
    score_method=['icl', 'clc', 'bic'])

# h.GMM("0", 40, use_scs_channel=[0], 
#     score_method=['icl', 'clc', 'bic'])
# h.GMM("1", 40, use_scs_channel=[1], 
#     score_method=['icl', 'clc', 'bic'])

#%%
name = ["0"]
f, ax = plt.subplots(1,2, sharey=False)
ax[0].plot(h.result[name[0]].score['icl'])
ax[1].plot(h.result[name[0]].score['nllh'])

#%%
# Use ADF + ABF
# import qem.classes as ps
# volume1 = f['adf']['volume'][:]
# volume2 = f['abf']['volume'][:]

# h = ps.GaussianMixtureModelObject(np.concatenate([volume1, volume2]), coordinate)
# h.GMM(40, criteria=['nllh', 'icl', 'clc', 'bic'], pos_init=(1,0))
# h.plot_criteria(['icl', 'clc', 'bic'])
# n_component = 31
# h.plot_histogram(n_component=n_component)
# h.plot_thickness(n_component=n_component, show_component=[20])


# %%
