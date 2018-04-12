#This has very similar functionality to PCA.py except includes all the plotting stuff which is hard to isolate. Thus, I made PCA.py so that it can be used in conjunction with other code (e.g. sPCA.py) and not have the plotting crap get in the way.

import numpy as np
import PCA as pca_py
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
# Get Wavelimits - (a,b) pairs to grab for each spectra, 296-540nm~free of telluric lines (http://diglib.nso.edu/flux)
def get_wavelims(plot_choice, binsize=250, center=4600):
    if plot_choice == 0:
        #---------standard range
        wavelims = [(4000,5700)]  #single range
        #wavelims = [((4000,4200),(4900,5700))] #multiple subranges in pca
    elif plot_choice == 1:
        #---------wavelength bins for fixed bin width
        n = (5750-4000)/bin_size + 1
        wavelims = np.linspace(4000,5750,n)
        wavelims = zip(wavelims[0:-1],wavelims[1:])
    elif plot_choice == 2:
        #---------increasing bin widths about a central value
        bin_widths = np.logspace(0,2.5,7,dtype=int)
        wavelims = []
        for b in bin_widths:
            wavelims.append((center-b,center+b))
    return wavelims

########## Perform PCA ##########
def do_PCA(wavelimits, dir, n_analyze, plot_choice, n_components=2, save=1):

    # naming extension
    ext = ''
    if n_analyze > 0:
        ext = '_n%d'%n_analyze

    try:
        X_pc = np.load('%sX_%dpcs%s.npy'%(dir,n_components,ext))
        RV = np.load('%sRV%s.npy'%(dir,ext))
        dates = np.load('%sdates%s.npy'%(dir,ext))
        #print "Successfully loaded PC-projected spectra"
    except:
        # get X/wavelengths, prune bad wavelengths
        X, wavelengths, dates, RV = pca_py.get_X(dir, wavelimits, save)
        
        std_cutoff = 0.1                                    #near empty rows, std skyrokets
        means, stds = np.mean(X, axis=0), np.std(X, axis=0)
        good = np.where((stds>0)&(stds<std_cutoff))[0]      #remove bad/empty rows
        X, wavelengths, means, stds = X[:,good], wavelengths[good], means[good], stds[good]
        
        # Normalize - don't standardize (divide by stds) since all dimensions are of same type.
        # Doing so would wrongly re-weight the importance of small/large feature variations.
        Xs = X - means
        
        # Do PCA
        pca = PCA(n_components=n_components)
        X_pc = pca.fit_transform(Xs)
        print "%d PCs explain %f of the variance for wavelimits:."%(n_components, np.sum(pca.explained_variance_ratio_)), wavelimits
        
        # Save projected spectra
        if save == 1:
            np.save('%sX_%dpcs%s.npy'%(dir,n_components,ext), X_pc)
        
        # inverse transform reconstructs original spectra
        if plot_choice == 0:
            normed, pname = 0, ''
            # Reconstruct spectra using principal components
            Xs_hat = np.dot(X_pc[:,:n_components], pca.components_[:n_components,:])
            for i_s in range(3):
                if normed == 1:
                    orig, reconstruct = Xs[i_s], Xs_hat[i_s]
                    pname='normed'
                else:
                    orig, reconstruct = Xs[i_s]*stds + means, Xs_hat[i_s]*stds + means
#                orig, reconstruct = scaler.inverse_transform(Xs[i_s]), scaler.inverse_transform(Xs_hat[i_s])
                plt.plot(wavelengths, orig, label='original')      #original signal
                plt.plot(wavelengths, reconstruct, label='reconstructed',alpha=0.8)  #reduced PCA
                plt.plot(wavelengths, np.abs(orig - reconstruct),alpha=0.5,label='abs(orig - rec)')
                plt.legend(loc='lower left', fontsize=8)
                plt.xlabel('wavelength')
                plt.ylabel('normalized flux')
                plt.xlim([4402,4410])
                plt.ylim([0,0.7])
                plt.title('%d PCs explain %f of the variance.'%(n_components, np.sum(pca.explained_variance_ratio_)))
                plt.savefig('output/reconstruct_images/reconstructed_pc%d_i%d_%s_zoom.png'%(n_components,i_s,pname))
                plt.clf()
            plt.plot(wavelengths,stds)
            plt.ylabel('std')
            plt.xlabel('wavelength')
            plt.savefig('output/reconstruct_images/std_v_wavelength.png')
            plt.clf()

    return X_pc, dates, RV, np.sum(pca.explained_variance_ratio_)

########## Main Routine ##########
if __name__ == '__main__':
    # data details
    #dir = "data/2012-2013/"           # data directory
    dir = "data/"
    n_analyze = -1          # number of files to analyze. -1 means all in the directory
    n_PCA_components = [2,10,50,100]    # number of pca components
    save = 0                # 1 = save files
    
    # plot_choice: 0=single range, 1=vs wavelength bins, 2=vs increasing bin size (about center)
    plot_choice = 0
    bin_size, center = 250, 4600
    wavelims = get_wavelims(plot_choice, bin_size, center)

    # Main Loop
    for pcs in n_PCA_components:
        x_plot, y_plot = [], []
        for WL in wavelims:
            # Get PC spectra
            X_pc, dates, RV, ev = do_PCA(WL, dir, n_analyze, plot_choice, pcs, save)
        
            if plot_choice == 1:
                x_plot.append(WL[0])
                y_plot.append(ev)
            elif plot_choice == 2:
                x_plot.append(WL[1] - WL[0])
                y_plot.append(ev)
    
        if plot_choice > 0:
            plt.plot(x_plot, y_plot, '--o', label='%d PCs'%pcs)

    ####################################################
    # Plotting
    if plot_choice == 1:
        plt.xlabel('wavelength (bin size=%d)'%bin_size)
        plt.ylabel('explained variance')
        plt.legend()
        plt.savefig('output/images/wavelength_vs_variance_%dbinsize.png'%bin_size)
        plt.show()
    elif plot_choice == 2:
        plt.xlabel('bin size (center=%d)'%center)
        plt.ylabel('explained variance')
        plt.legend()
        plt.savefig('output/images/binsize_vs_variance_%dcenter.png'%center)
        plt.show()


