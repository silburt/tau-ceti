import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
def get_X(dir, n_analyze, wavelim_a, wavelim_b, plot_lines=0, save=1):
    # get files
    readme = open(glob.glob("%sREADME_*.txt"%dir)[0],"r").readlines()
    fits_files = glob.glob("%s*.fits"%dir)

    ext = ''
    if n_analyze > 0:
        fits_files = fits_files[0:n_analyze]
        ext = '_n%d'%n_analyze

    try:
        X = np.load('%sX%s.npy'%(dir,ext))
        dates = np.load('%sdates%s.npy'%(dir,ext))
        print "Successfully loaded spectra"
    except:
        print "Couldn't load spectra, extracting from scratch..."
        X, dates = [], []
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
            
            # grab wavelength ranges of interest (i.e. leave out earth-atmosphere wavelengths)
            flux_patches = get_spectral_patches(wave, flux, wavelim_a, wavelim_b)
            
            # sort by date
            # probably not neccesary? PCA doesn't make use of temporal/spatial correlations
            #X, dates = sort_obs(flux_patches, date, X, dates)
            X.append(flux_patches)
            dates.append(date)

        X, dates = np.array(X), np.array(dates)
        if save == 1:
            np.save('%sX%s.npy'%(dir,ext),X)
            np.save('%sdates%s.npy'%(dir,ext),dates)

    return X, dates

########## Main Routine ##########
if __name__ == '__main__':
    # arguments
    dir = "data/"           # data directory
    n_analyze = -1          # number of files to analyze. -1 means all in the directory
    n_PCA_components = 5    # number of pca components
    wavelim_a = [4000]      # wavelengths to grab, [a,b] are limit pairs
    wavelim_b = [6000]
    
    # Get spectra
    X, dates = get_X(dir, n_analyze, wavelim_a, wavelim_b)
    
    # Normalize spectra
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)    #s for scaled

    # Do PCA
    print "Finished getting data, performing PCA"
    pca = PCA(n_components=n_PCA_components)
    pca.fit(Xs)
    print(pca.explained_variance_ratio_)
    
    # Reconstructed spectra using principal components
    n_reconstruct = 2
    Xs_hat = np.dot(pca.transform(Xs)[:,:n_reconstruct], pca.components_[:n_reconstruct,:])

    # plotting stuff
    # inverse transform reconstructs original spectra
    plt.plot(scaler.inverse_transform(Xs[10]))
    plt.plot(scaler.inverse_transform(Xs_hat[10]))
    plt.xlim([350,450])
    plt.show()


