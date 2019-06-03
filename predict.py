import numpy as np
from sgdml.predict import GDMLPredict

def load_model(model_path):
    '''
    This function is called once during initialisation. Any one-time loading
    that needs to be done should be done here. If nothing needs loading, this
    function still needs to exist but can just return immediately.
    
    Parameters:
      -model_path: 
          string of the path to the model file
    '''

    try:
        model=np.load(model_path)
    except:
        print("Could not load model file. Please ensure it is of the right format (.npz)")
        sys.exit(2)

    try:
        global gdml
        gdml=GDMLPredict(model)
    except:
        print("Unable to read GDML model file.")
        sys.exit(2)

def predict(cluster_R):
    '''
    Function used to predict forces of data chunks. Pray pay attention to the dimensions.
    The values predicted here are equivalent to those present in the ['F'] key of the data.
    The input is an array containing the spatial information of every sample in a single 
    cluster.
    The output is an array containing the output values (f.e. forces) for every sample in 
    the cluster (same order as the input). 

    The output values are to be given in a 1D single array, so that the function return:
    cluster_F= [[F11,F12,...,F1M],[F21,F22,...,F2M],...,[FN1,FN2,...,FNM]] 
    for a cluster containing N samples and a model output of a total of M values. 
    
    
    Paramters:
        - cluster_R: numpy array containing sample positions. 
                     Dimensions are (n_samples_in_cluster,n_atoms,n_dimensions)
        
    Returns:
        - cluster_F: numpy array containing predicted forces.   
                     Dimensions are (n_samples_in_cluster,n_atoms*n_dimensions)
    '''
    
    global gdml
    
    n_samples,n_atoms,n_dim=cluster_R.shape
    cluster_R=np.reshape(cluster_R,(n_samples,n_atoms*n_dim))  
    
    
    _,cluster_F=gdml.predict(cluster_R)

    return cluster_F


def predict_energies(cluster_R):
    '''
    See predict. Note:
    - The values predicted here are equivalent to those present in the ['E'] key of the data.
    - This function is used instead of predict (above) if the -n argument is used when calling errors.py
    '''


    global gdml
    
    n_samples,n_atoms,n_dim=cluster_R.shape
    cluster_R=np.reshape(cluster_R,(n_samples,n_atoms*n_dim))  
    
    
    cluster_E,_=gdml.predict(cluster_R)

    return cluster_E




