import numpy as np
from scipy.spatial.distance import pdist

def toDistance(a):
    shape=a.shape
    try:
        dim=shape[2]
    except:
        print("toDistance: wrong dimensions")
        return
    if shape[1]<2:
        print("not enough atoms")
        return

    y=[]

    for i in range(len(a)): ##goes through samples
        y.append(pdist(a[i]))

    y=np.array(y)
    return y
 

def r_to_desc(r):
	n_atoms = r.shape[0]
	return 1. / toDistance(r)
