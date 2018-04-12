# Plots the weights of each PC

import PCA as pca_py
import numpy as np
import matplotlib.pyplot as plt

dir = "data/"           # data directory
n_pcs = 8
wavelims = (4000,5700)

# do PCA
X, V, Z, Xs_hat, X_hat, wavelengths, ev, n = pca_py.do_PCA(dir, wavelims, n_pcs)

# plot
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12,8))
for i in range(2):
    ax1.plot(wavelengths, V[i,:], ',', alpha=1-0.1*i, label='pc %d'%(i))
for i in range(2):
    ax2.plot(wavelengths, V[i+2,:], ',', alpha=1-0.1*i, label='pc %d'%(i+2))
for i in range(2):
    ax3.plot(wavelengths, V[i+4,:], ',', alpha=1-0.1*i, label='pc %d'%(i+4))
for i in range(2):
    ax4.plot(wavelengths, V[i+6,:], ',', alpha=1-0.1*i, label='pc %d'%(i+6))
ax1.legend(fontsize=8, numpoints=1)
ax2.legend(fontsize=8, numpoints=1)
ax3.legend(fontsize=8, numpoints=1)
ax4.legend(fontsize=8, numpoints=1)
ax4.set_xlabel('wavelength')
ax4.set_ylabel('eigenvector values')
#ax1.set_ylim([-0.02,0.02])
ax1.set_title('explained variance = %f'%ev)
plt.savefig('output/pca_visualize_%dcomponents.png'%n_pcs)
