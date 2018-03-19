import numpy as np
from octree import *
#Added Functions

def local_PCA(points):
    """
    Compute the eigenvalues and the eigenvectors of the covariance matrix of the points cloud.
    Input : points size N x 3
    output: 
    -eigenvalues size 3 x 1
    -eigenvectors size 3 x 3 the columns of this matrix are the eigenvectors
    """
    eigenvalues = None
    eigenvectors = None
    # It is different from the expected covariance matrix so
    # the eigenvalues are different from the example.
    # to have the same values , we can multiply Sigma by N.
    if(points.shape[0] < 2):
        points = points.reshape(1, 3)
        Sigma = points.T @ points
    else:
        Sigma = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    ind = np.argsort(eigenvalues)[::-1]
    
    return eigenvalues[ind], eigenvectors[:, ind]



def compute_new_features(eigenvals, eigenvects):
    """
    compute a list of features that include :
    -sphericity
    -linearity
    -verticality
    -planarity
    -entropy
    -anisotropy
    -surface variation
    -sum
    -Omnivariance
    output: matrix X size N x d
    
    """
    sigma = eigenvals.sum(axis=1).reshape(-1,1)
    o = (np.log(eigenvals[:,0]+0.0001)+np.log(eigenvals[:,1]+0.0001)+np.log(eigenvals[:,2]+0.0001))/3
    o = np.exp(o).reshape(-1,1)
    a = (eigenvals[:,0]-eigenvals[:,2])/(eigenvals[:,0]+0.0001)
    a = a.reshape(-1,1)
    p = (eigenvals[:,1]+eigenvals[:,2])/(eigenvals[:,0]+0.0001)
    p = p.reshape(-1,1)
    l = (eigenvals[:,0]+eigenvals[:,1])/(eigenvals[:,0]+0.0001)
    l = l.reshape(-1,1)
    sv = eigenvals[:,2]/(eigenvals.sum(axis=1)+0.001)
    sv = sv.reshape(-1,1)
    s = eigenvals[:,2]/(eigenvals[:,0]+0.0001)
    s = s.reshape(-1,1)
    v = 1 - np.abs(eigenvects[:,2,2])
    v = v.reshape(-1,1)
    e = eigenvals * np.log(eigenvals+0.0001)
    e = e.sum(axis=1)
    e = e.reshape(-1,1)
    
    return np.hstack((sigma,o,a,p,l,sv,s,v,e))


def compute_covariance_feat(points):
    eigenvals, eigenvects = local_PCA(points)
    return compute_new_features(eigenvals.reshape(1,3), eigenvects.reshape(1,3,3))

def compute_ppl_feat(points):
    tree = Octree(4, points)
    min_max = points.max(axis=0)-points.min(axis=0)
    std = points.std()
    phi = tree.get_phi()
    return np.concatenate((tree.divide()[1:], min_max, [std], phi[1], phi[2], phi[3], phi[4]))



