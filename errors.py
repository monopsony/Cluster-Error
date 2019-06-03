import numpy as np
import sys,getopt,os,shutil
from sklearn.metrics import mean_squared_error
import cluster
from descri import r_to_desc
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))+"/"

#used to make .xyz files
_z_str_to_z_dict = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,'Uuq':114,'Uuh':116}
_z_to_z_str_dict = {v: k for k, v in _z_str_to_z_dict.iteritems()}


def print_usage():
    print('''
Calculates errors based on model and data file.
python error.py -m <model_file> -d <data_file> [-c <clusters_file>] [-e]

Data must contain atomic positions, forces and energies (data['R'],data['F'] and data['E']).
Prediction error done on forces (data['R']) unless -e argument is called.


Re-draw graph of an already computed cluster error storage file:
python error.py -g <storage_file>

Optional arguments:
-h    shows this help message 
-e    calculates energy prediction error instead of forces
-r    resets parameter file to default (then exits)
-c    allows the user to input a .npy file of cluster indices manually
    ''')

def reset_para_file():
    para_file='''

#cluster parameters
number_of_spacial_clusters=10      #how many clusters to create during the initial agglomerative clustering step
number_of_energy_clusters=5        #how many clusters to separate every cluster of previous step into
initial_spatial_data_points=30000  #determines how many points are used for the initial agglomerative clustering step
                                   #if a memory problem/overflow occurs, try reducing this value
                                   #otherwise, a larger number of initial data points will lead to better clustering

#graph parameters
x_axis_label="Cluster index"
y_axis_label=r'Force prediction error $(kcal/mol/\AA)^2$'
y_axis_label_energies=r'Energy prediction error $(kcal/mol/)^2$'  #used when -n option is enabled, i.e. comparing data['E'] energy predictions
fontsize1=30 #used for axis labels
fontsize2=30*0.85 #used for tick labels
linewidth1=5 #used for axes
horizontal_line=True #if True, includes a horizontal line to show the average error 
linewidth2=3 #used for horizontal "average error" line
horizontal_line_color="green" 
size_in_inches=(18.5,11) #graph size
transparent_background=True 
include_population=True  #indicates the cluster population for every cluster 
total_population=False  #if include_population is True, the population will instead be shown as a total population curve
order_by_energy=False     #if true, the graph will order the clusters by their average energy rather than error
reverse_order=False  #if False, orders from lowest error/energy to highest (left to right), reversed otherwise
    
    '''
        
    if os.path.exists(path+"para.py"):
        os.remove(path+"para.py")
    f=open(path+"para.py","w")
    f.write(para_file)
    f.close()
    
def parse_arguments(argv):
    '''
    This function parses through the arguments and acts accordingly.
    
    Parameters:
        -argv:
            array containing information on input arguments, argv=sys.argv[1:]
            
    Arguments:
        -m:
            Saves corresponding var to model_path
            
        -d:
            Saves corresponding var to dataset_path
            
        -h:
            Prints the help message (using print_usage) then exits.
            
        -e:    
            sets predict_energies=True. Used to compare energies rather than forces (data['E'] rather than data['F'])
            
        -r:
            resets the para.py file to the default values then exits.

        -g:
            enters graph_only mode. Saves corresponding var to mse_path.

        -c:
            gives the option to input your own .npy file of cluster indices. Saves corresponding var to cluster_path.

            
    Returns:
        model_path,dataset_path,graph_only,mse_path,no_graph,cluster_path
        
        -model_path,dataset_path,cluster_path:
            string corresponding to path to model, dataset and cluster file respectively
        
        -graph_only:
            boolean indicating whether to enter graph-only mode. Needs mean_squared_error.txt file
            from prior errors.py run.
            
        -mse_path:
            string corresponding to path to mean_squared_error.txt file
            
        -no_graph:
            boolean indicating whether to make a graph. Ignored when in graph_only mode.  
    '''
    
    model_path,dataset_path=None,None
    try:
        opts,args=getopt.getopt(argv,"hrm:d:g:c:e")
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    
    #defaults
    no_graph,cluster_path,predict_energies=False,False,False
    
    for opt,arg in opts:
        if opt=='-h':
            print_usage()
            sys.exit()
            
        elif opt=="-m":
            model_path=arg
            
        elif opt=="-d":
            dataset_path=arg
    
        elif opt=="-c":
            cluster_path=arg or nil
            if not cluster_path:
                print("No cluster path given after -c option. Use the -h argument for help.")
                sys.exit(2)
            elif not os.path.exists(cluster_path):
                print("No cluster indices file found under path "+cluster_path)
                sys.exit(2)

        elif opt=="-g":
            print("Graph only mode:")
            graph_dir_path=arg
            if not graph_dir_path:
                print("Path to mean-squared error textfile required.")
            else:
                return None,None,True,graph_dir_path,False,False

        #DEPRECATED
        #elif opt=="-e":
        #    no_graph=True  

        elif opt=="-r":
            reset_para_file()
            print("Reset parameter file")
            sys.exit()
    
        elif opt=="-e":
            predict_energies=True

    if not model_path:
        print("No model path given. Use the -h argument for help.")
        sys.exit(2)
    elif not os.path.exists(model_path):
        print("No model file found under path "+model_path)
        sys.exit(2)
        
    if not dataset_path:
        print("No dataset path given. Use the -h argument for help.")
        sys.exit(2)
    elif not os.path.exists(dataset_path):
        print("No dataset file found under path "+dataset_path)
        sys.exit(2)
    
    return model_path,dataset_path,False,None,no_graph,cluster_path,predict_energies

def create_storage_directory(model_path,dataset_path,cluster_path,predict_energies):
    '''
    Creates the directory in which to store the results.
    Directory will be {name_of_model_file}_{name_of_dataset_file} excluding extensions
    and parent directories. 
    
    If a directory of the same name already exists, it will be overwritten.
    
    Parameters:
        -model_path,dataset_path,cluster_path:
            string corresponding to the path of the model, dataset and cluster file respectively
            
    Returns:
        -storage_dir:
            string corresponding to the path to the newly created storage directory
    '''

    path_separator=((os.name=='nt') and "\\") or  "/" 

    dir_name="Default"+path_separator
    model_name=os.path.splitext(os.path.basename(model_path))[0]
    dataset_name=os.path.splitext(os.path.basename(dataset_path))[0]
    if cluster_path:
        cluster_name="_c_"+os.path.splitext(os.path.basename(cluster_path))[0]
    else:
        cluster_name=''
    energy_indicator=(predict_energies and "_energy") or ""
    dir_name=model_name+"_"+dataset_name+cluster_name+energy_indicator
    
    storage_dir=path+path_separator+"storage"+path_separator+dir_name

    path_index,i='',0
    while os.path.exists(storage_dir+path_index):
        i=i+1
        path_index='_'+str(i)
        if i>1000:
            break
    storage_dir=storage_dir+path_index+path_separator

    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    
    os.makedirs(storage_dir)
    
    return storage_dir

#Assumes that the atoms in each molecule are in the same order.
def read_concat_ext_xyz(f):
    '''
    Reads the content of an xyz file and saves it into arrays.
    
    Paramters:
        -f:
            xyz file to be parsed through
    Returns:
        (R,z,E,F)
        
        -R:
            numpy array containing positions of atoms in each sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
            
        -F:
            numpy array containing forces of atoms in each sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
            
        -E:
            numpy array containing energies of atoms in each sample
            Dimensions: (n_samples,1)
           
        -z:
            numpy array containing atom type in order
            Dimensions: (n_atoms)
    '''

    n_atoms = None

    R,z,E,F = [],[],[],[]
    for i,line in enumerate(f):
        line = line.strip()
        if not n_atoms:
            n_atoms = int(line)

        file_i, line_i = divmod(i, n_atoms+2)

        if line_i == 1:
            E.append(float(line))

        cols = line.split()
        if line_i >= 2:
            R.append(map(float,cols[1:4]))
            if file_i == 0: # first molecule
                z.append(_z_str_to_z_dict[cols[0]])
            F.append(map(float,cols[4:7]))

    R = np.array(R).reshape(-1,n_atoms,3)
    z = np.array(z)
    E = np.array(E)
    F = np.array(F).reshape(-1,n_atoms,3)

    f.close()
    return (R,z,E,F)

def save_clusters_xyz(storage_dir,dataset,cluster_indices,mse):
    '''
    Saves all clusters individually as xyz files into the storage directory.
    Names are simply "cluster{cluster number}.xyz". Cluster number goes in ascending order 
    of mean squared error;  Cluster50.xyz has the largest error.
    
    Parameters:
        -storage_dir:
            string corresponding to the path to the storage directory
        
        -dataset:
            variable containing the initially loaded dataset
            
        -cluster_indices:
            array containing clusters and their contained indices corresponding to dataset
            
        -mse:
            array containing mean squared error for each sample in every cluster
    '''

    print("Saving clusters...")
    try:
        atoms=dataset["z"]
        nAtoms=len(atoms)
    except:
        print("Dataset missing information on atom names. Could not produce cluster xyz files.")
        return
    
    R=dataset["R"]
    
    n_clusters=len(cluster_indices)
    sys.stdout.write( "\r[0/{}] done".format(n_clusters) )
    sys.stdout.flush()
    for i in range(n_clusters):
        cl=cluster_indices[i]
        name=("{}Cluster{}.xyz").format(storage_dir,i+1)
        file=open(name,"w")

        for k in range(len(cl)):
            indR=R[cl[k]]
            file.write(("{}\n".format(nAtoms)))
            file.write(("#Atomic units, mean squared error: {} \n").format(mse[i][k]))
            
            for j in range(nAtoms):
                file.write( ("{} {} {} {} \n").format(  _z_to_z_str_dict[atoms[j]],indR[j][0],indR[j][1],indR[j][2]))
                
        file.close()
        sys.stdout.write( "\r[{}/{}] done".format(i+1,n_clusters) )
        sys.stdout.flush()
    
    print("\n")
    
def load_dataset(dataset_path):
    '''
    Loads the dataset file (.xyz or .npz) and saves it into a dictionary. 
    
    Parameters:
        -dataset_path:
            string corresponding to the path to the dataset file
        
    Returns:
        -data:
            dictionary containing all necessary information on the dataset. 
            Needs to include:
            
            -"R":
                numpy array of positions for all samples
                Dimensions: (n_samples,n_atoms,n_dimensions)
                
            -"F":
                numpy array of forces for all samples
                Dimensions: (n_samples,n_atoms,n_dimensions)
            
            -"E":
                numpy array of energies for all samples
                Dimensions: (n_samples,1)
                
            -"z":
                array containing list of atom type in order 
                Dimensions: (n_atoms)
                Used to crate cluster xyz files
                
    '''
    ext=os.path.splitext(dataset_path)[-1]
    
    #xyz file
    if ext==".xyz":
        try:
            file=open(dataset_path)
            dat=read_concat_ext_xyz(file)
            data={ 'R':np.array(dat[0]),'z':dat[1],'E':np.reshape( dat[2] , (len(dat[2]),1) ),'F':np.array(dat[3]) }
        except getopt.GetoptError as err:
            print(err)
            return False
    #npz file        
    elif ext==".npz":
        try:
            data=np.load(dataset_path)
        except:
            return False
            
    return data
   
def calculate_errors(dataset,cluster_indices,predict_energies):
    """
    Calculates the mean squared error between forces predicted by the model
    and forces given in the dataset file for each cluster individually.
    
    Parameters:
        -dataset:
            dataset containing all necessary information (R, F and E)
        
        -cluster_indices:
            array containing indices for each cluster

        -predict_energies:
            boolean set True by the -e argument. 
            If True, computes energy prediction error rather than forces.

    Returns:
        -mse:
            numpy array of mean squared errors for each cluster, sorted by descending order
            
        -cluster_indices:
            numpy array of cluster_indices but sorted by descending order of cluster mean squared errors
            
        -sample_errors:
            numpy array of mean squared errors for each sample within every cluster. Clusters sorted 
            by descending order of cluster mean squared error
    """
    
    print("Calculating errors for each cluster...")

    
    #helping variables
    n_clusters=len(cluster_indices)
    R,F,E=dataset["R"],dataset["F"],dataset["E"]
    mse=[]
    sample_errors=[]

    #loop through clusters
    #predict results for each cluster
    #calculate error, save in list
    sys.stdout.write( "\r[0/{}] done".format(n_clusters) )
    sys.stdout.flush()
    for i in range(n_clusters):
        cind=cluster_indices[i] #cluster indices
        cr,cf,ce=R[cind],F[cind],E[cind] #cluster R 
        shape=cr.shape
        n_samples,n_atoms,n_dim=shape[0],shape[1],False
        if len(shape)>2:
            n_dim=shape[2]
        
        if predict_energies:
            cf=np.array(ce)
            cf_pred=predict.predict_energies(cr)

        else:
            #reshaping
            if n_dim:
                cf=np.reshape(cf,(n_samples,n_atoms*n_dim))  
            else:
                cf=np.reshape(cf,(n_samples,n_atoms)) 
            cf_pred=predict.predict(cr)

        err=(cf-cf_pred)**2
        sample_errors.append(err.mean(axis=1))
        mse.append(err.mean())

        #print out
        sys.stdout.write( "\r[{}/{}] done".format(i+1,n_clusters) )
        sys.stdout.flush()
     
    print("")   
    #order the cluster_indices etc
    sorted_ind=np.argsort(mse)
    mse=np.array(mse)
    cluster_indices=np.array(cluster_indices)
    sample_errors=np.array(sample_errors)

    return mse[sorted_ind],cluster_indices[sorted_ind],sample_errors[sorted_ind]
    
def error_graph(mse,dir,cluster_indices,E,predict_energies):

    '''
    Generates and saves the graph of the mean squared error for each cluster. File name is "graph.png" 
    and is saved inside of the storage directory. This function is a complete mess and needs a remake
    to be readable at all, so enter with caution.
    
    Some parameters for the graph generation can be found in the para.py file.
    
    Parameters:
        -mse:
            numpy array containing mean squared errors for each cluster as given by calculate_errors()
            
        -dir:
            string corresponding to path to storage directory
            
        -cluster_indices
            numpy array containing a list of indices of for every cluster
            
        -E
            numpy array containing the energy of every sample of the original dataset
    '''
    
    #by default the cluster_indices and mse are ordered by error due to calculate_errors
    if para.order_by_energy:
        cE=[]
        for i in range(len(cluster_indices)):
            cE.append( np.average(E[cluster_indices[i]]) )
        cE=np.array(cE)
        
        order=np.argsort(cE)
        cluster_indices=cluster_indices[order]
        mse=mse[order]
        
    if True:
        pop,tot=[],0
        
        #calculate total population
        for c in cluster_indices:
            tot=tot+len(c)
        
        if para.total_population:
            for c in cluster_indices:
                if len(pop)==0:
                    pop.append(len(c))
                else:
                    pop.append(pop[-1]+len(c))
        else:
            for c in cluster_indices:
                pop.append(len(c))
            

        pop=np.array(pop).astype(float)/np.max(pop)

    #x axis
    x=np.arange(len(mse))+1
    if para.reverse_order:
        x=np.flip(x)
    
    #helping variables
    min,max,avg=np.min(mse),np.max(mse),np.average(mse)
    real_avg=0

    #calculate true avg
    poptot=np.sum(pop)
    real_avg=  np.sum(np.array(pop)*np.array(mse))/poptot
    

    med=(max+min)/2
    #xticks=[]
    #yticks=[min,avg,med,max]
    #xlabels=[]
    #ylabels=["{:.1f}".format(min),"{:.1f}".format(avg),"{:.1f}".format(med),"{:.1f}".format(max)]
    fs=para.fontsize1
    fs2=para.fontsize2
    lw=para.linewidth1
    lw2=para.linewidth2
    
    #create figure
    f,ax1=plt.subplots()
    ax1.bar(x,mse)
    if para.include_population:
        if para.total_population:
            pop=pop*max  #rescale to make sense in the graph (was normalised to 0-1 at creation) 
        else:
            pop=pop*med  #id
        ax1.step(x,pop,c="orange",where="mid",linewidth=lw)

        #ax1.text(0,avg-(max-min)*0.04,"Relative population",fontsize=fs2,color='orange')
        ax1.text(-8.5*len(pop)/50.,med,r'Relative cluster size',fontsize=fs,verticalalignment='center',horizontalalignment='center',color='orange',rotation=90)
        

    #ax1.set_xticks(xticks)
    #ax1.set_xticklabels(xlabels,fontsize=fs)
    #ax1.set_yticks(yticks)
    #ax1.set_yticklabels(ylabels,fontsize=fs2)
    plt.xlabel(para.x_axis_label,labelpad=10,fontsize=fs)
    plt.ylabel((predict_energies and para.y_axis_label_energies) or para.y_axis_label,fontsize=fs)
    ax1.tick_params(axis='both',labelsize=fs2)    

    for i in ["top","right"]:
        ax1.spines[i].set_visible(False)
    for i in ["left","bottom"]:
        ax1.spines[i].set_linewidth(lw)

    ax1.xaxis.set_tick_params(width=0)
    ax1.yaxis.set_tick_params(width=0)

    if para.horizontal_line:
        ax1.axhline(real_avg,linewidth=lw2,color=para.horizontal_line_color)
    ax1.text(0,real_avg+(max-min)*0.01,"average mean squared error",fontsize=fs2,color=para.horizontal_line_color)

   
    f.set_size_inches(para.size_in_inches)
    f.savefig(dir+"graph.png",transparent=para.transparent_background)
    print("Graph saved at {}".format(dir+"graph.png"))

def save_mse_file(storage_dir,mse):
    '''
    Saves the mean squared error for each cluster into a text file inside the storage directory.
    This file can be used for the graph-only mode [-g] to generate a new graph.
    
    Parameters:
        -storage_dir:
            string corresponding to the path to the storage directory
            
        -mse:
            numpy array of mean squared error for each cluster
    
    '''
    
    if os.path.exists(storage_dir+"mean_squared_error.txt"):
        os.remove(storage_dir+"mean_squared_error.txt")
    
    f=open(storage_dir+"mean_squared_error.txt","w")
    f.write("#Cluster number \t  mean squared error\n")
    for i in range(len(mse)):
        f.write( ("{}\t{}\n").format(i+1,mse[i]))
    f.close()

def save_dataset_energies(storage_dir,E):
    '''
    Saves the energies of the original dataset as a numpy array. This is only required
    to enable graph mode usage when the order_by_energy parameter is set to true.
    
    Parameters:
        -storage_dir:
            string corresponding to the path to the storage directory
            
        -E:
            numpy array of energies of the original dataset
    
    '''
    
    if os.path.exists(storage_dir+"dataset_energies.npy"):
        os.remove(storage_dir+"dataset_energies.npy")
    np.save(storage_dir+"dataset_energies.npy",E)

if __name__=="__main__":
    
    path_separator=((os.name=='nt') and "\\") or  "/" 
            
    #load arguments 
    model_path,dataset_path,graph_only,graph_dir_path,no_graph,cluster_path,predict_energies=parse_arguments(sys.argv[1:])

    #import parameter file
    import para

    #enter graph-only mode
    #exits after execution
    if graph_only:
        storage_directory=graph_dir_path+path_separator
        try:
            mse=np.loadtxt(storage_directory+"mean_squared_error.txt")
        except Exception as e:
            print("Could not load mse text file at "+storage_directory+"mean_squared_error.txt. Please ensure the path is correct and the file is of the correct type.")
            print(e)
            sys.exit(2)
               
        try:
            cluster_indices=np.load(storage_directory+"cluster_indices_all.npy")
        except Exception as e:
            print("Could not load cluster_indices file at "+storage_directory+"cluster_indices_all.npy. Please ensure the path is correct and the file is of the correct type.")
            print(e)
            sys.exit(2)
            
        try:
            E=np.load(storage_directory+"dataset_energies.npy")
        except Exception as e:
            print("Could not load mse text file at "+storage_directory+"dataset_energies.npy. Please ensure the path is correct and the file is of the correct type.")
            print(e)
            sys.exit(2)
        
        error_graph(mse[:,1],storage_directory,cluster_indices,E,predict_energies)
        sys.exit()
    
    #try to load the dataset
    dataset=None
    dataset=load_dataset(dataset_path)
    if not dataset:
        print("Could not load dataset file. Please ensure it is of the right format (.npz or .xyz)")
        sys.exit(2)

    #try to load model
    import predict
    predict.load_model(model_path)
    
    #prepare data 
    try:
        R=r_to_desc(dataset["R"])
        E=np.array(dataset["E"])
        F=np.array(dataset["F"])
    except getopt.GetoptError as err:
        print("Unable to find necessary data in data set. Data set must contain ['R'],['F'] and ['E'] keys corresponding to spatial data, forces and energies.")
        print(err)
        sys.exit(2)

    #create storage directory and save path to it for further functions    
    storage_dir=create_storage_directory(model_path,dataset_path,cluster_path,predict_energies)

    #try to load cluster_indices if given or
    #cluster the data, return indices of each cluster
    if cluster_path:
        try:
            cluster_indices=np.load(cluster_path)
        except:
            print("Could not load cluster indices. Please make sure it is of the right format (.npz)")
    else: 
        cluster_indices=cluster.cluster(R,E)
    
    #calculate errors for every cluster/sample within each cluster
    mse,cluster_indices,samples_mse=calculate_errors(dataset,cluster_indices,predict_energies)

    #save cluster indices
    if os.path.exists(storage_dir+"cluster_indices_all.npy"):
        os.remove(storage_dir+"cluster_indices_all.npy")
    np.save(storage_dir+"cluster_indices_all.npy",cluster_indices)

    #save cluster mean squared error as text   
    save_mse_file(storage_dir,mse)
    
    #save clusters into the storage directory
    save_clusters_xyz(storage_dir,dataset,cluster_indices,samples_mse)

    #save database energy for further graph_mode usage
    save_dataset_energies(storage_dir,E)

    #make and store the graph unless argument [-e] was present
    if not no_graph:
        error_graph(mse,storage_dir,cluster_indices,E,predict_energies)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
