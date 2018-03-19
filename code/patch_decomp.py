

"""
MVA project
Course : Nuage de points
Author : Sofiane Horache

Here we are going to find some functions to process data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def pc_to_patches(xyz, prec=1):
    """
    assign a id to a point corresponding to a square patch
    input:
    xyz vector n x 3
    prec : float precision in meter
    output:
    dictionnary where
    keys are id of patches
    values are index of points cloud
    
    """
    new_xyz = xyz.copy()/prec
    xyz_fl = pd.DataFrame(np.floor(new_xyz), columns=['x', 'y', 'z'])
    gb = xyz_fl.groupby(['x', 'y', 'z']).groups

    return dict([(i, np.array(k[1])) for i, k in enumerate(gb.items())])


def patches_to_id(patches, n):
    """
    transform a dictionary of index into a list that indicate
    for each point the
    id of its patches
    input:
    patches, a dictionary that contains for each label the list of points
    n, size of the point cloud
    output:
    array of size n
    """
    id_patches = np.zeros(n)
    for ind, lis in patches.items():
        id_patches[lis] = ind
    return id_patches


def id_to_dic(id_patches):
    """
    from list to dictionary
    input:
    a list
    output:
    dictionary that contains id of points for each patches
    """
    keys = np.unique(id_patches)
    index = np.arange(len(id_patches))
    out = dict([(k,index[id_patches == k]) for k in keys])
    return out
 
    
def labelize_patch(id_patches, y):
    one_hot = y.copy()
    lab = LabelBinarizer()
    one_hot = lab.fit_transform(one_hot)
    df = pd.DataFrame(np.hstack((one_hot, id_patches.reshape(-1,1))), columns=list(np.unique(y))+['patches'])
    gb = df.groupby(['patches']).sum()
    label_per_patch = gb.apply(np.argmax, axis=1)
    return label_per_patch.astype(int)

