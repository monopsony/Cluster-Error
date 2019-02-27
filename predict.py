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
    Function used to predict forces of data chunks.
    
    
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

    