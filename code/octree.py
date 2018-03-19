"""
MVA project
Course : Nuage de points
Author : Sofiane Horache

Here, We define our octree class to get the lod(level of detail)
"""

import numpy as np


class OctNode:
    def __init__(self, data, children, min_ptx, max_ptx):
        self.children = children
        self.data = data
        self.min_ptx = min_ptx
        self.max_ptx = max_ptx
        
    def count_data(self):
        return len(self.data)

    
class Octree:
    def __init__(self, nb_lod, pts):
        self.lod = nb_lod
        self.nb_cube_per_lod = np.zeros(nb_lod+1)
        self.nb_cube_per_lod[0] = 1
        self.phi = dict()
        self.phi[0] = np.array([1])
        for lo in range(nb_lod):
            self.phi[lo+1] = np.zeros(8**(lo+1))
        self.root = OctNode(pts, [], pts.min(axis=0), pts.max(axis=0))
    
    def divide_(self, node, lod, id_p):
        """
        compute the descriptors at each levels
        recursive function
        """
        if node.count_data() == 0 or lod >= self.lod:
            return None
        else:
            center = 0.5 * (node.min_ptx+node.max_ptx)
            is_ge = np.zeros(node.data.shape[0])
            for ind, c in enumerate(center):
                is_ge = is_ge + (node.data[:, ind] > c) * 2**ind
            for i in range(8):
                # for each little square
                new_data = node.data[is_ge == i]
                # define the border
                min_ptx = np.zeros(3)
                max_ptx = np.zeros(3)
                is_one = i
                for j in range(3):
                    if(is_one % 2 == 0):
                        min_ptx[j] = node.min_ptx[j]
                        max_ptx[j] = center[j]
                    else:
                        min_ptx[j] = center[j]
                        max_ptx[j] = node.max_ptx[j]
                    is_one = is_one // 2
                    
                # add the new children to the node
                newNode = OctNode(new_data, [], min_ptx, max_ptx)
                node.children.append(newNode)
                # update the nb of cube per lod
                if(len(new_data) > 0):
                    self.nb_cube_per_lod[lod+1] += 1
                    self.phi[lod+1][id_p+i*8**(lod)] = 1
                
                # apply the same principle to the children
                self.divide_(newNode, lod+1, id_p+i*8**(lod))
                
    def divide(self):
        """
        compute the descriptors and 
        return the nb of cube per LOD
        """
        self.divide_(self.root, 0, 0)
        return self.nb_cube_per_lod

    def get_phi(self):
        """
        return descriptors called phi. can be used to define a kernel.
        """
        return self.phi
            
        
        
