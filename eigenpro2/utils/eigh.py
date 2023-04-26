'''Utility functions for performing fast SVD.'''
# import scipy.linalg as linalg
import torch
from math import sqrt


def nystrom_kernel_eigh(samples, kernel_fn, top_q):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.

    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """

    n_sample, _ = samples.shape
    samples_ = samples #.cpu()
    kmat = kernel_fn(samples_, samples_)
    scaled_kmat = kmat / n_sample
    vals, vecs = torch.lobpcg(scaled_kmat, min(top_q+1, n_sample//3))
    # vals, vecs = linalg.eigh(scaled_kmat,
    #                         eigvals=(n_sample - top_q, n_sample - 1))
    #eigvals = torch.from_numpy(vals).flip(0)
    #eigvecs = torch.from_numpy(vecs).fliplr()/sqrt(n_sample)
    beta = kmat.diag().max()

    # return eigvals.float(), eigvecs.float(), beta
    return vals, vecs/math.sqrt(n_sample), beta
