from PCA import *
import numpy as np
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA

########## Main Routine ##########
if __name__ == '__main__':
    # data details
    dir = "data/"           # data directory
    n_pcs = 2
    n_spcs = 4
    alpha = [0.5,1,2,5,7]    #sparsity parameter. Higher=more sparse
    wavelims = (4000,5700)

    # do PCA
    X, Z, Xs_hat, X_hat, wavelengths = do_PCA(dir, wavelims, n_pcs)

    # get residuals
    X_residual = X - X_hat
    means, stds = np.mean(X_residual, axis=0), np.std(X_residual, axis=0)
    Xs_residual = (X_residual - means)/stds

    # sparse PCA
    print "starting sparse PCA"
    for a in alpha:
        spca = MiniBatchSparsePCA(n_components=n_spcs, alpha=a)
        spca.fit(Xs_residual)
        for i in range(n_spcs):
            plt.plot(spca.components[i,:], label='sparse pc %d'%i)
        plt.legend(fontsize=8)
        plt.savefig('output/sparse_pca_alpha=%.2f.png'%a)
        plt.clf()
        print "completed alpha=%.2f"%a

