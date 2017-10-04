import numpy as np
from sklearn.decomposition import PCA
import sys
import glob
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import os
import tarfile
from datetime import datetime

########### get tar file and extract RV correction ##########
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

########## Grab Wavelengths of Interest ##########
def get_spectral_patches(wave, flux, lim_a, lim_b):
    patches = np.zeros(0)
    for i in range(len(lim_a)):
        a, b = lim_a[i], lim_b[i]
        patches = np.concatenate((patches,flux[(wave>=a)&(wave<b)]))
    return np.array(patches)

########## Sort Observations ########## - maybe need a faster way?
def sort_obs(flux_patches, date, X, dates):
    for i in range(len(X)):
        if date < dates[i]:
            X.insert(i, flux_patches)
            dates.insert(i, date)
            return X, dates
    X.append(flux_patches)
    dates.append(date)
    return X, dates

########## Get array of spectral features through time ##########
def get_X(dir, n_analyze, wavelim_a, wavelim_b, plot_lines=0):
    # get files
    readme = open(glob.glob("%sREADME_*.txt"%dir)[0],"r").readlines()
    fits_files = glob.glob("%s*.fits"%dir)
    
    X, dates = [], []
    timeformat = "%Y-%m-%dT%H:%M:%S.%f"
    for fits in fits_files[0:n_analyze]:
        hdulist = pyfits.open(fits)
        
        # get to the data part (in extension 1)
        scidata = hdulist[1].data
        wave = scidata[0][0]
        flux = scidata[0][1]
        err = scidata[0][2]
        
        # red/blue shift due to RV
        tar_asson1 = hdulist[0].header['ASSON1']
        RV_c = get_star_RV(tar_asson1, readme)
        wave -= RV_c*wave
        
        # grab wavelength ranges of interest (i.e. leave out earth-atmosphere wavelengths)
        flux_patches = get_spectral_patches(wave, flux, wavelim_a, wavelim_b)
        
        # sort by date
        date = datetime.strptime(hdulist[0].header['DATE-OBS'],timeformat)
        X, dates = sort_obs(flux_patches, date, X, dates)

    return np.array(X), np.array(dates)

########## Main Routine ##########
if __name__ == '__main__':
    # arguments
    dir = "data/"           # data directory
    n_analyze = 20          # number of fits file to analyze
    n_PCA_components = 5    # number of pca components
    wavelim_a = [5000]      # wavelengths to grab, [a,b] are limit pairs
    wavelim_b = [6000]
    
    # Main calcs
    X, dates = get_X(dir, n_analyze, wavelim_a, wavelim_b)

    print "Finished getting data, performing PCA"
    pca = PCA(n_components=n_PCA_components)
    pca.fit(X)
    print(pca.explained_variance_ratio_)


