import sys
import glob
from astropy.io import fits as pyfits
import numpy as np
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

########## get line ratio for given obs ##########
def get_line_ratio(wave, flux, lines, base_offsets, lw):
    line_time = []
    for i in range(len(lines)):
        l, b = lines[i], base_offsets[i]
        line = np.sum(flux[(wave>l-lw)&(wave<l+lw)])
        base = np.sum(flux[(wave>l-b-lw)&(wave<l-b+lw)])
        line_time.append(line/base)
    return line_time

########## Sort Observations ##########
def sort_obs(line_ratio, date, line_ratios, dates):
    for i in range(len(line_ratios)):
        if date < dates[i]:
            line_ratios.insert(i, line_ratio)
            dates.insert(i, date)
            return line_ratios, dates
    line_ratios.append(line_ratio)
    dates.append(date)
    return line_ratios, dates

########## Main Loop ##########
def extract_line_timeseries(dir, n_analyze, lines, base_offsets, linewidth, line_names):
    # get files
    readme = open(glob.glob("%sREADME_*.txt"%dir)[0],"r").readlines()
    fits_files = glob.glob("%s*.fits"%dir)
    
    line_ratios, dates = [], []
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
        
        # get line info
        line_ratio = get_line_ratio(wave, flux, lines, base_offsets, linewidth)
        date = datetime.strptime(hdulist[0].header['DATE-OBS'],timeformat)
        line_ratios, dates = sort_obs(line_ratio, date, line_ratios, dates)

        plot_lines = 0
        if plot_lines == 1:
            #line location
            line_i = 2      #line to plot
            plt_line, plt_base = lines[line_i], base_offsets[line_i]
            plt_flux = np.min(flux[(wave>plt_line-linewidth)&(wave<plt_line+linewidth)])

            #plots
            plt.plot(wave, flux)
            plt.plot([plt_line-linewidth,plt_line+linewidth], [plt_flux,plt_flux], label='line')
            plt.plot([plt_line-plt_base-linewidth,plt_line-plt_base+linewidth], [plt_flux,plt_flux], label='baseline')
            plt.xlim([plt_line-1.5*np.abs(plt_base),plt_line+1.5*np.abs(plt_base)])
            
            #dont convert x-axis to scientific notation.
            ax = plt.gca()
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            plt.xlabel('Wavelength (angstrom)')
            plt.ylabel('Flux (adu)')
            plt.title('%s, lambda=%.2f angstroms'%(line_names[line_i],lines[line_i]))
            loc = 'lower right'
            if plt_base < 0:
                loc = 'lower left'
            plt.legend(loc=loc)
            plt.savefig('output/images/%s.png'%os.path.basename(fits).split('.fits')[0])
            plt.clf()

    return np.array(line_ratios), np.array(dates)

########## Main Routine ##########
if __name__ == '__main__':
    dir = "data/"
    n_analyze = 10

    #line info - units of angstroms
    line_names = ['Ca I','Na I D','Na I D']
    lines = [6572.795,5889.95,5895.92]
    base_offsets = [2,2,2]
    linewidth = 0.5

    line_ratios, dates = extract_line_timeseries(dir, n_analyze, lines, base_offsets, linewidth, line_names)
    
    plt.plot(dates, line_ratios[:,0], 'o-')
    plt.xticks(rotation=30)
    plt.ylabel('line/base ratio')
    plt.show()

################## NOTES ####################
# http://docs.astropy.org/en/stable/io/fits/index.html
# A.S. Some potentially useful commands
'''
    print dir(hdulist)
    print hdulist[0].header[0:50]       #things like RA/Dec info
    print hdulist[0].header['TITLE']
    print hdulist.info()
'''
    # print column information
    #print hdulist.info()        #general overview of what's inside
    #print hdulist[1].columns

# RV->wavelength shift - https://www.e-education.psu.edu/astro801/content/l4_p7.html

#extra
#    fits_basename = os.path.basename(fits)
#        if fits_basename in l:
#            if ".tar" in readme[i-1]:
#                i_tar = i-1
#            elif ".tar" in readme[i+1]:
#                i_tar = i+1
#            else:
#                print "couldnt find corresponding tar file"
