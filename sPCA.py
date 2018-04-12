from PCA import *
import numpy as np
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA

########## Main Routine ##########
if __name__ == '__main__':
    # data details
    dir = "data/"           # data directory
    n_pcs = 6
    n_spcs = 15
    alpha = [1e-5,1e-4,1e-3,1e-2,1e-1]    #sparsity parameter. Higher=more sparse
    wavelims = (4000,5700)

    # do PCA
    X, V, Z, Xs_hat, X_hat, wavelengths, ev, n = do_PCA(dir, wavelims, n_pcs)

    # get residuals
    X_residual = X - X_hat
    means, stds = np.mean(X_residual, axis=0), np.std(X_residual, axis=0)
    #Xs_residual = (X_residual - means)/stds
    Xs_residual = X_residual - means

    # sparse PCA
    print "starting sparse PCA"
    for a in alpha:
        spca = MiniBatchSparsePCA(n_components=n_spcs, alpha=a)
        spca.fit(Xs_residual)

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        for i in range(len(V)):
            ax1.plot(wavelengths, V[i,:], ',', alpha=1-0.1*i, label='pc %d'%i)
        for i in range(5):
            ax2.plot(wavelengths,spca.components_[i,:], ',', alpha=1-0.1*i, label='sparse pc %d'%i)
        for i in range(5):
            ax3.plot(wavelengths,spca.components_[i+5,:], ',', alpha=1-0.1*i, label='sparse pc %d'%i)
        for i in range(5):
            ax4.plot(wavelengths,spca.components_[i+10,:], ',', alpha=1-0.1*i, label='sparse pc %d'%i)
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        ax4.set_xlabel('wavelength')
        ax1.set_ylabel('eigenvector values')
        ax2.set_ylabel('eigenvector values')
        plt.savefig('output/sparse_pca_alpha=%.2e.png'%a)
        plt.clf()
        print "completed alpha=%.2e"%a

'''
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=True)
    for i in range(n_pcs):
        ax1.plot(wavelengths, V[n_pcs-i-1,:], ',', alpha=1-0.1*i, label='pc %d'%i)
    #    for i in range(5):
    #        ax2.plot(wavelengths, V[i+5,:], ',', alpha=1-0.1*i, label='pc %d'%(i+5))
    #    for i in range(5):
    #        ax3.plot(wavelengths, V[i+10,:], ',', alpha=1-0.1*i, label='pc %d'%(i+10))
    #    for i in range(5):
    #        ax4.plot(wavelengths, V[i+15,:], ',', alpha=1-0.1*i, label='pc %d'%(i+15))
    ax1.legend(fontsize=8, numpoints=1)
    ax4.set_xlabel('wavelength')
    ax4.set_ylim([-0.02,0.02])
    ax1.set_ylabel('eigenvector values')
    ax2.set_ylabel('eigenvector values')
    ax1.set_title('explained variance = %f'%ev)
    plt.savefig('output/pca_visualize_%dcomponents.png'%n_pcs)
'''

# for plotting just the PCs (for group meeting)
#    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#    for i in range(2):
#        ax1.plot(wavelengths, V[i,:], ',', alpha=1-0.1*i, label='pc %d'%i)
#    for i in range(2,n_pcs):
#        ax2.plot(wavelengths, V[i,:], ',', alpha=1-0.1*i, label='pc %d'%(i))
#    ax1.legend(fontsize=10)
#    ax2.legend(fontsize=10)
#    ax2.set_xlabel('wavelength')
#    ax1.set_ylabel('eigenvector values')
#    ax2.set_ylabel('eigenvector values')
#    ax1.set_title('explained variance of %d PCs = %f'%(n_pcs,ev))
#    plt.savefig('output/pca.png')
