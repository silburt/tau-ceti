# April 18th, 2018 - Plots the weights of each PC

import PCA as pca_py
import numpy as np
import matplotlib.pyplot as plt

#dir = "data/2012-2013/"           # data directory
dir = "data/"
n_pcs = 8
#wavelims = (4000,5700)              # clean range
wavelims = (3800,6850)              # full range

# do PCA
X, V, Z, Xs_hat, X_hat, wavelengths, ev, n_spec = pca_py.do_PCA(dir, wavelims, n_pcs)

# plot PCs
f, ax = plt.subplots(n_pcs, 1, sharex=True, figsize=(12,8))
plt.subplots_adjust(hspace=0)
ax[n_pcs-1].set_xlabel('wavelength (angstroms)')
for i in range(0, n_pcs):
    ax[i].plot(wavelengths, V[i,:], ',')
    ax[i].set_ylabel('PC %d'%(i+1))

ax[0].set_title('explained variance = %f, n_spectra=%d'%(ev, n_spec))
plt.savefig('output/pca_visualize_%dcomponents_n%d.png'%(n_pcs, n_spec))
plt.close()

# plot mean Spectra
X_mean = np.mean(X, axis=0)
plt.figure(figsize=(13,8))
plt.plot(wavelengths, X_mean)
plt.xlabel('wavelength (angstroms)')
plt.ylabel('flux')
plt.savefig('output/mean_spectra_n%d.png'%n_spec)
plt.close()

# draw random spectra
n_draw = 30
np.random.seed(12)   #42 is a good choice locally
rs = np.arange(len(X))
np.random.shuffle(rs)
rs = rs[0:n_draw]

# plot spectra sub-regions
lims = [(5000,5025),(6525,6600)]
for lim in lims:
    l1, l2 = lim
    plt.figure(figsize=(13,8))
    for i in range(0,n_draw):
        plt.plot(wavelengths, X[rs[i]], 'k,')
    plt.plot(wavelengths, X_mean, linewidth=1)
    plt.xlim([l1, l2])
    plt.xlabel('wavelength (angstroms)')
    plt.ylabel('flux')
    plt.savefig('output/mean_spectra_n%d_%d-%d.png'%(n_spec, l1, l2))
    plt.close()

# plot PCA sub-regions
for lim in lims:
    l1, l2 = lim
    # calculate explained variance for region
    #1-  [ sum_j sum_i (X_i,j - reconstruct_with_k_components_i)^2 ] /  [ sum_j sum_i (X_i,j - mean_X_i)^2 ]
    lim_index = (wavelengths>=l1)&(wavelengths<=l2)
    X_, X_hat_, X_mean_ = X[:,lim_index], X_hat[:,lim_index], X_mean[lim_index]
    ex_var = 1 - np.sum((X_ - X_hat_)**2) / np.sum((X_ - X_mean_)**2)
    # plot
    f, ax = plt.subplots(n_pcs, 1, sharex=True, figsize=(13,8))
    plt.subplots_adjust(hspace=0)
    ax[n_pcs-1].set_xlabel('wavelength (angstroms)')
    for i in range(0, n_pcs):
        ax[i].plot(wavelengths, V[i,:], linewidth=1)
        ax[i].set_ylabel('PC %d'%(i+1))
    ax[0].set_title('*local* explained variance = %f, n_spectra=%d'%(ex_var, n_spec))
    ax[0].set_xlim([l1,l2])
    plt.savefig('output/pca_visualize_%dcomponents_n%d_%d-%d.png'%(n_pcs, n_spec, l1, l2))
    plt.close()
