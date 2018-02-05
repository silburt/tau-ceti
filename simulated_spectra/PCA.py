#This just has the bare bones for doing PCA, no plotting or endless options.
#An important thread for my understanding is:
#https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
#and I follow their notation.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import glob
import pandas as pd
import os

def sort_files(files):
    phase = []
    for f in files:
        phase.append(float(f.split('_')[-1].split('.csv')[0]))
    return [x for _, x in sorted(zip(phase,files), key=lambda pair: pair[0])]

# Get flux array for corresponding wavelengths of interest
def get_wave_range(X, wave, lims):
    XX = []
    for i in range(len(X)):
        Xi = X[i]
        Xconcat = np.zeros(0)
        if type(lims[0]) == tuple:
            for l in lims:  #grab multiple sub-ranges for single spectra
                Xconcat = np.concatenate((Xconcat,Xi[(wave>=l[0])&(wave<l[1])]))
        else:
            Xconcat = np.concatenate((Xconcat,Xi[(wave>=lims[0])&(wave<lims[1])]))
        XX.append(Xconcat)

    return np.array(XX)

########## Get array of spectral features through time ##########
def get_X(dir, wavelimits):
    
    try:
        X = np.load('files/flux.npy')
        wavelength = np.load('files/wavelength.npy')
        print('Successfully loaded files')
    except:
        print('Couldnt load files, building from scratch')
        # get files
        files = sort_files(glob.glob("%s*.csv"%dir))
        X = []
        for f in files:
            data = pd.read_csv(f)
            flux, wavelength = data['Intensity'].values, data['Wavelength'].values
            
            # normalize spectra by continuum region
            norm = np.sum(flux) / len(flux)
            flux /= norm
            
            X.append(flux)
        np.save('files/flux.npy',np.asarray(X))
        np.save('files/wavelength.npy',wavelength)

    # grab wavelength ranges of interest (i.e. leave out earth-atmosphere wavelengths)
    X = get_wave_range(X, wavelength, wavelimits)

    return X, wavelength

########## Perform PCA ##########
def do_PCA(dir, wavelimits, n_components, reconstruct=1):
    
    # get X/wavelengths, prune bad wavelengths
    X, wavelengths = get_X(dir, wavelimits)
    means, stds = np.mean(X, axis=0), np.std(X, axis=0)
    #Xs = (X - means)/stds                           #normalize
    Xs = X/stds
    
    # Do PCA
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xs)                       #PCA projections (scores/loadings)
    V = pca.components_                             #eigenvectors (directions of maximum variance)
    print("%d PCs explain %f of the variance for wavelimits:."%(n_components, np.sum(pca.explained_variance_ratio_)), wavelimits)
    
    Xs_hat, X_hat = None, None
    if reconstruct == 1:
        # Reconstruct spectra using n pcs (X_hat = XVV^T = ZV^T)
        Xs_hat = np.dot(Z, V)                           #normalized
        X_hat = Xs_hat*stds + means                     #unnormalized

    return X, V, Z, Xs_hat, X_hat, wavelengths, np.sum(pca.explained_variance_ratio_)
