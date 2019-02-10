This small program is designed to calculate the prediction errors that a given trained model makes across different geometries of the target molecule. 

The program requires two files from the user in order to function: the trained model, to be given as a npz file, and the dataset to test the model on, which can be given in an xyz or npz format. 

To perform the test, simply run the ‘errors.py’ file, preferably using a python 2.7 interpreter. The following parameters are required: ‘-d’ followed by a path to the desired testing dataset, and ‘-m’ followed by a path to the model file. 

Usage example: 
python errors.py -d dataset.xyz -m model.npz

Furthermore, an input file ‘para.py’ is provided, allowing the user to alter some of the parameters used for the clusterisation. If needed, the parameter file can be reset back to the default values using the -r operation.

After successfully running errors.py, a folder named ‘{name of model file}_{name of dataset file}’ will appear in the '/storage' directory, containing a png file of a column graph of the average error on every cluster, as well as an xyz file for all data points in each respective cluster in order of the highest error (with ‘Cluster1.xyz’ having the largest average error).