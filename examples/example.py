#%%
import qem
import matplotlib.pyplot as plt
import numpy as np

legacyStatSTEM = qem.io.read_legacyInputStatSTEM('Example_PtIr.mat')
image = legacyStatSTEM['input']['obs']
dx = legacyStatSTEM['input']['dx']
inputStatSTEM = legacyStatSTEM['input']
outputStatSTEM = legacyStatSTEM['output']

input_coordinates = inputStatSTEM['coordinates']
output_coordinates = outputStatSTEM['coordinates']

from qem.fitting import ImageModelFitting

model=ImageModelFitting(image, pixel_size=dx)
# switch the first and second columns of input_coordinates
input_coordinates = input_coordinates[:,[1,0]]
model.import_coordinates(coordinates=input_coordinates)
model.same_width = True
model.device = 'cuda'

model.optimize(lr=0.001,rel_tol=1e-5, max_iter=1000)