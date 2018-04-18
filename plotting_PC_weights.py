# April 18th, 2018 - Plots the weights of each PC

import PCA as pca_py
import numpy as np
import matplotlib.pyplot as plt

dir = "data/2012-2013/"           # data directory
#dir = "data/"
n_pcs = 8
#wavelims = (4000,5700)              # clean range
wavelims = (3800,6850)              # full range

# do PCA
X, V, Z, Xs_hat, X_hat, wavelengths, ev, n_spec = pca_py.do_PCA(dir, wavelims, n_pcs)

# plot PCs
f, ax = plt.subplots(n_pcs, 1, sharex=True, figsize=(12,8))
plt.subplots_adjust(hspace=0)
ax[0].set_ylabel('PC 0 and 1')
ax[n_pcs-1].set_xlabel('wavelength (angstroms)')
for i in range(0, n_pcs):
    ax[i].plot(wavelengths, V[i,:], ',')
    ax[i].set_ylabel('PC %d'%(i+1))

ax[0].set_title('explained variance = %f, n_spectra=%d'%(ev, n_spec))
plt.savefig('output/pca_visualize_%dcomponents.png'%n_pcs)
plt.close()

# plot mean Spectra
X_mean = np.mean(X, axis=0)
plt.figure(figsize=(13,8))
plt.plot(wavelengths, X_mean)
plt.xlabel('wavelength (angstroms)')
plt.ylabel('flux')
plt.savefig('output/mean_spectra_n%d.png'%n_spec)
plt.close()

# busy region
l1, l2 = 6000, 6500
plt.figure(figsize=(13,8))
plt.plot(wavelengths, X_mean)
plt.xlim([l1, l2])
plt.xlabel('wavelength (angstroms)')
plt.ylabel('flux')
plt.savefig('output/mean_spectra_n%d_%d-%d.png'%(n_spec, l1, l2))
plt.close()

# quiet region
l1, l2 = 4000, 5000
plt.figure(figsize=(13,8))
plt.plot(wavelengths, X_mean)
plt.xlim([l1, l2])
plt.xlabel('wavelength (angstroms)')
plt.ylabel('flux')
plt.savefig('output/mean_spectra_n%d_%d-%d.png'%(n_spec, l1, l2))
plt.close()
