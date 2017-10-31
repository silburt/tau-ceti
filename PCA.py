#This just has the bare bones for doing PCA, no plotting or endless options.
#An important thread for my understanding is:
#https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
#and I follow their notation.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import glob
from astropy.io import fits as pyfits
import os
import tarfile
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

########### Helper Functions ##########
# get tar file and extract RV correction
def get_star_RV(tar_asson1, readme):
    feature = 'HIERARCH ESO DRS CCF RV'
    for l in readme:
        if tar_asson1 in l:
            tar_string = "%sADP.%s.tar"%(dir,l.split("ADP.")[1].split(".tar")[0])
            try:
                tar = tarfile.open(tar_string)
                ext_tar = tar.extractfile(tar.getmembers()[1])
                RV = pyfits.open(ext_tar)   #ccf_G2_A.fits is 2nd index
                RV_c = RV[0].header[feature] / 299792.458   #RV / speed of light
            except:
                print "couldnt load %s."%tar_string
                return 0
            break
    return RV_c

# Get flux array for corresponding wavelengths of interest
def get_wave_range(X, wavelengths, lims):
    XX = []
    for i in range(len(X)):
        Xi, wave = X[i], wavelengths[i]
        Xconcat = np.zeros(0)
        if type(lims[0]) == tuple:
            for l in lims:  #grab multiple sub-ranges for single spectra
                Xconcat = np.concatenate((Xconcat,Xi[(wave>=l[0])&(wave<l[1])]))
        else:
            Xconcat = np.concatenate((Xconcat,Xi[(wave>=lims[0])&(wave<lims[1])]))
        XX.append(Xconcat)

    # grab wavelengths
    wave_concat = np.zeros(0)
    if type(lims[0]) == tuple:
        for l in lims:  #grab multiple sub-ranges for single spectra
            wave_concat = np.concatenate((wave_concat,wave[(wave>=l[0])&(wave<l[1])]))
    else:
        wave_concat = np.concatenate((wave_concat,wave[(wave>=lims[0])&(wave<lims[1])]))

    return np.array(XX), wave_concat

########## Get array of spectral features through time ##########
def get_X(dir, wavelimits, save=1):
    
    # get files
    readme = open(glob.glob("%sREADME_*.txt"%dir)[0],"r").readlines()
    fits_files = glob.glob("%s*.fits"%dir)

    try:
        X = np.load('%sX.npy'%dir)
        wavelengths = np.load('%swavelengths.npy'%dir)
        dates = np.load('%sdates.npy'%dir)
        RV = np.load('%sRV.npy'%dir)
    except:
        print "Couldn't load spectra, extracting from scratch..."
        X, wavelengths, RV, dates = [], [], [], []
        timeformat = "%Y-%m-%dT%H:%M:%S.%f"
        for fits in fits_files:
            hdulist = pyfits.open(fits)
            date = datetime.strptime(hdulist[0].header['DATE-OBS'],timeformat)
            
            # get to the data part (in extension 1)
            scidata = hdulist[1].data
            wave = scidata[0][0]
            flux = scidata[0][1]
            err = scidata[0][2]
            
            # red/blue shift due to RV
            tar_asson1 = hdulist[0].header['ASSON1']
            RV_c = get_star_RV(tar_asson1, readme)
            wave -= RV_c*wave
            
            # normalize spectra by continuum region
            norm = np.sum(flux) / len(flux)
            flux /= norm
        
            X.append(flux)
            dates.append(date)
            wavelengths.append(wave)
            RV.append(RV_c)     #Maybe this is the wrong RV?

        # sort dates- https://stackoverflow.com/questions/20533335/sorting-lists-by-datetime-in-python
        zipped = zip(X, wavelengths, dates, RV)
        zipped = sorted(zipped, key=lambda t: t[2])
        X, wavelengths, dates, RV = zip(*zipped)

        # save
        X, wavelengths, dates, RV = np.array(X), np.array(wavelengths), np.array(dates), np.array(RV)
        if save == 1:
            np.save('%sX.npy'%dir,X)
            np.save('%sdates.npy'%dir,dates)
            np.save('%sRV.npy'%dir,RV)
            np.save('%swavelengths.npy'%dir,wavelengths)

    # grab wavelength ranges of interest (i.e. leave out earth-atmosphere wavelengths)
    X, wavelengths = get_wave_range(X, wavelengths, wavelimits)

    return X, wavelengths, dates, RV

########## Perform PCA ##########
def do_PCA(dir, wavelimits, n_components, save=1):
    
    # get X/wavelengths, prune bad wavelengths
    X, wavelengths, dates, RV = get_X(dir, wavelimits, save)

    std_cutoff = 0.1                                    #near empty rows, std skyrokets
    means, stds = np.mean(X, axis=0), np.std(X, axis=0)
    good = np.where((stds>0)&(stds<std_cutoff))[0]      #remove bad/empty rows
    X, wavelengths, means, stds = X[:,good], wavelengths[good], means[good], stds[good]
    
    try:
        Z = np.load('%sZ_%dpcs.npy'%(dir,n_components))
        Xs_hat = np.load('%sXshat_%dpcs.npy'%(dir,n_components))
        X_hat = np.load('%sXhat_%dpcs.npy'%(dir,n_components))
    except:
        # Do PCA
        pca = PCA(n_components=n_components)
        Xs = (X - means)/stds                           #normalize->mu=0,std=1
        Z = pca.fit_transform(Xs)                       #PCA projections ("scores")
        print "%d PCs explain %f of the variance for wavelimits:."%(n_components, np.sum(pca.explained_variance_ratio_)), wavelimits

        # Reconstruct spectra using n pcs (X_hat = XVV^T)
        Xs_hat = np.dot(Z, pca.components_)             #normalized
        X_hat = Xs_hat*stds + means                     #unnormalized

        # Save projected spectra
        if save == 1:
            np.save('%sZ_%dpcs.npy'%(dir,n_components),Z)
            np.save('%sXshat_%dpcs.npy'%(dir,n_components),Xs_hat)
            np.save('%sXhat_%dpcs.npy'%(dir,n_components),X_hat)

    return X, Z, Xs_hat, X_hat, wavelengths

########## Main Routine ##########
if __name__ == '__main__':
    # data details
    dir = "data/"             # data directory
    n_PCA_components = [1,2]    # number of pca components
    
    wavelims = [(4000,5700)]  #single range
    #wavelims = [((4000,4200),(4900,5700))] #multiple subranges in pca

    # Main Loop
    for pcs in n_PCA_components:
        for WL in wavelims:
            X, Z, Xs_hat, X_hat, wavelengths = do_PCA(dir, WL, pcs)



