# Plots the weights of each PC

import PCA as pca_py
import numpy as np
import matplotlib.pyplot as plt

dir = "data/2012-2013/"           # data directory
n_pcs = 8
wavelims = (4000,5700)

# do PCA
X, V, Z, Xs_hat, X_hat, wavelengths, ev, n_spec = pca_py.do_PCA(dir, wavelims, n_pcs)

# plot
f, ax = plt.subplots(n_pcs-1, 1, sharex=True, figsize=(12,8))
plt.subplots_adjust(hspace=0)
ax[0].set_ylabel('PC 0 and 1')
ax[n_pcs-2].set_xlabel('wavelength')
for i in range(2):
    ax[0].plot(wavelengths, V[i,:], ',', alpha=1-0.8*i, label='pc %d'%(i))
for i in range(1, n_pcs-1):
    ax[i].plot(wavelengths, V[i+1,:], ',')
    ax[i].set_ylabel('PC %d'%(i+1))

ax[0].set_title('explained variance = %f, n_spectra=%d'%(ev, n_spec))
plt.savefig('output/pca_visualize_%dcomponents.png'%n_pcs)
