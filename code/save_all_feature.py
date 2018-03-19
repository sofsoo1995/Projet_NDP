"""
Here we will compute features and save it
"""

# library used
import numpy as np
import pandas as pd
from utils.ply import *
from patch_decomp import *
from descriptor import * 
import matplotlib.pyplot as plt
from octree import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score
import itertools
import time


# load data
list_filepath = ["../data/GT_Madame1_3", "../data/Cassette_GT"]
for filepath in list_filepath:
    print("for :"+filepath)
    data = read_ply(filepath+".ply")
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    label = data['class']
    print("Generating patches")
    patches = pc_to_patches(xyz, 1) # very long !!!
    id_patches = patches_to_id(patches, xyz.shape[0])
    write_ply(filepath+"_patched.ply",np.hstack((xyz,
                                                 id_patches.reshape(-1,1),
                                                 label.reshape(-1, 1))),
              ['x', 'y', 'z', 'patches', 'class'])
    print("Done")
    print("Computing the label for each patch")
    label_per_patch = labelize_patch(id_patches, y)
    new_label = np.zeros(y.shape[0])
    new_label = label_per_patch[id_patches]
    write_ply(filepath+"_patch_label.ply",np.hstack((xyz,
                                                         id_patches.reshape(-1,1),
                                                         new_label.reshape(-1,1))),
              ['x', 'y', 'z', 'patches', 'new_label'])
    
    print("Done")
    print("compute features:")
    data = read_ply(filepath+"_patch_label.ply")
    df = pd.DataFrame(np.vstack((data['x'], data['y'], data['z'], data['patches'])).T,
                      columns=['x', 'y', 'z', 'patches'])
    new_y = data['new_label']
    # Function to compute the features
    cov = lambda col:compute_covariance_feat(col.loc[:,('x','y', 'z')].values)
    ppl = lambda col:compute_ppl_feat(col.loc[:,('x','y', 'z')].values)
    print("compute cov coeff...")
    start = time.time()
    X_cov = np.vstack((df.groupby('patches').apply(cov).values))
    print("Done in {0}".format(time.time()-start))
    print('compute ppl coef...')
    start = time.time()
    X_ppl = np.vstack((df.groupby('patches').apply(ppl).values))
    print("Done in {0}".format(time.time()-start))
    # That's why we save the data
    np.save(filepath+'_feature_cov', X_cov)
    np.save(filepath+'_feature_ppl', X_ppl)