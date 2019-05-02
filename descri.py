import numpy as np
from scipy.spatial.distance import pdist

def toDistance(R):
    '''
    This function takes a numpy array containing positions and returns it as distances.
    
    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
            
    Returns:
        -y:
            numpy array containing distances for every atom in every sample
            Dimensions: (n:samples,n_atoms*(n_atoms-1)/2)
    '''
    
    shape=R.shape
    try:
        dim=shape[2]
    except:
        print("toDistance: wrong dimensions")
        return
    if shape[1]<2:
        print("not enough atoms")
        return

    y=[]

    for i in range(len(R)): ##goes through samples
        y.append(pdist(R[i]))

    y=np.array(y)
    return y
 

def r_to_desc(R):
    '''
    Returns the position array as an array of desired description.
    This description is solely used for clusterisation.
    Default is inverse distances.
    
    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
                    
    Returns:
        numpy array containing inverse distances for every atom in every sample
        Dimensions: (n:samples,n_atoms*(n_atoms-1)/2)
    '''
    return 1. / toDistance(R)
