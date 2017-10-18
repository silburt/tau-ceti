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
import george

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
#def get_spectral_patches(wave, flux, lims):
#    
#    patches = np.zeros(0)
#    for i in range(len(lims)):
#        a, b = lims[i]
#        patches = np.concatenate((patches,flux[(wave>=a)&(wave<b)]))
#    return np.array(patches), wave[(wave>=a)&(wave<b)]

# after arrays have already been grabbed and sorted.
def get_wave_range(X, wavelengths, lims):
    XX = []
    a, b = lims[0]
    for i in range(len(X)):
        arr, wave = X[i], wavelengths[i]
        XX.append(arr[(wave>=a)&(wave<b)])
    return np.array(XX)

########## Get array of spectral features through time ##########
def get_X(wavelims, dir, n_analyze, ext, save=1):
    
    # wavelength params - 2960-5400nm ~ free of telluric lines (http://diglib.nso.edu/flux)
    #continuum_norm = [4720,4810]    #normalize each spectra by continuum region
    continuum_norm = [4608,4609]
    
    # get files
    readme = open(glob.glob("%sREADME_*.txt"%dir)[0],"r").readlines()
    fits_files = glob.glob("%s*.fits"%dir)
    if n_analyze > 0:
        fits_files = fits_files[0:n_analyze]

    try:
        X = np.load('%sX%s.npy'%(dir,ext))
        wavelengths = np.load('%swavelengths%s.npy'%(dir,ext))
        dates = np.load('%sdates%s.npy'%(dir,ext))
        RV = np.load('%sRV%s.npy'%(dir,ext))
        print "Successfully loaded data"
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
            index = np.where((wave>=continuum_norm[0])&(wave<continuum_norm[1]))[0]
            norm = np.sum(flux[index]) / float(len(index))
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
            np.save('%sX%s.npy'%(dir,ext),X)
            np.save('%sdates%s.npy'%(dir,ext),dates)
            np.save('%sRV%s.npy'%(dir,ext),RV)
            np.save('%swavelengths%s.npy'%(dir,ext),wavelengths)

    # grab wavelength ranges of interest (i.e. leave out earth-atmosphere wavelengths)
    print "getting wavelength range"
    X = get_wave_range(X, wavelengths, wavelims)

    return X, dates, RV

########## Perform PCA ##########
def do_PCA(wavelims, dir, n_analyze, n_components=2, save=1, plot=0):

    # naming extension
    ext = ''
    if n_analyze > 0:
        ext = '_n%d'%n_analyze

    try:
        X_pc = np.load('%sX_pc%s.npy'%(dir,ext))
        RV = np.load('%sRV%s.npy'%(dir,ext))
        dates = np.load('%sdates%s.npy'%(dir,ext))
        #wavelengths = np.load('%swavelengths%s.npy'%(dir,ext))
        #print "Successfully loaded PC-projected spectra"
    except:
        #print "Couldn't load PC-projected spectra, generating..."
        X, dates, RV = get_X(wavelims, dir, n_analyze, ext, save)
        
        # Normalize to 0 mean, unit variance
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)    #np.dot(X, self.components_.T)
        
        # Do PCA
        pca = PCA(n_components=n_components)
        X_pc = pca.fit_transform(Xs)
        print "%d PCs explain %f of the variance:"%(n_PCA_components, np.sum(pca.explained_variance_ratio_)), pca.explained_variance_ratio_
        
        # Save projected spectra
        if save == 1:
            np.save('%sX_%dpcs%s.npy'%(dir,n_components,ext),X_pc)
        
        # plotting stuff
        # inverse transform reconstructs original spectra
        if plot == 1:
            i_s = 0
            # Reconstruct spectra using principal components
            Xs_hat = np.dot(pca.transform(Xs)[:,:n_components], pca.components_[:n_components,:])
            plt.plot(scaler.inverse_transform(Xs[i_s]))      #original signal
            plt.plot(scaler.inverse_transform(Xs_hat[i_s]))  #reduced PCA
            #plt.xlim([100000,100100])
            plt.yscale('log')
            plt.show()

    return X_pc, dates, RV

########## Main Routine ##########
if __name__ == '__main__':
    # data details
    dir = "data/"           # data directory
    n_analyze = -1          # number of files to analyze. -1 means all in the directory
    n_PCA_components = 97    # number of pca components
    save = 1                # 1 = save files
    
    # 2960-5400nm ~ free of telluric lines
    wavelims = [(4000,5700)]      # wavelengths to grab from each spectra, (a,b) are limit pairs
    #wavelims = np.linspace(4000,5750,17)
    #wavelims = zip(wavelims[0:-1],wavelims[1:])
    
    for WL in wavelims:
        # Get PC spectra
        print WL
        X_pc, dates, RV = do_PCA(wavelims, dir, n_analyze, n_PCA_components, save)
    
    # plotting
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(dates,X_pc[:,0])
    ax[0].plot(dates,X_pc[:,1])
    ax[1].plot(dates, RV)
    plt.xticks(rotation=30)
    plt.show()
    
