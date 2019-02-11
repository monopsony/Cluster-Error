import numpy as np
import sys,getopt,os,shutil
from sgdml.predict import GDMLPredict
from sklearn.metrics import mean_squared_error
import cluster
from descri import r_to_desc
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))+"/"

def print_usage():
    print('''
    Calculate errors based on model and database file:
    python error.py -m <model_file> -d <database_file> [-e]

    Generate graph based on mean-squared error text file:
    python error.py -g <mse_input_file>

    Optional arguments:
    -h    shows this help message
    -e    excludes graph from the outputs
    -r    resets parameter file to default (ignores all other arguments)
    
    ''')

def parse_arguments(argv):
    model_path,dataset_path=None,None
    try:
        opts,args=getopt.getopt(argv,"hrm:d:g:")
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    
    #defaults
    no_graph=False
    
    for opt,arg in opts:
        if opt=='-h':
            print_usage()
            sys.exit()
            
        elif opt=="-m":
            model_path=arg
            
        elif opt=="-d":
            dataset_path=arg
        
        elif opt=="-g":
            print("Graph only mode:")
            mse_path=arg
            if not mse_path:
                print("Path to mean-squared error textfile required.")
            else:
                return None,None,True,mse_path,False
        
        elif opt=="-e":
            no_graph=True

        elif opt=="-r":
            para_file='''
#cluster parameters
number_of_spacial_clusters=10
number_of_energy_clusters=5
initial_spatial_data_points=2500

#graph parameters
x_axis_label="Cluster number"
y_axis_label="Mean squared average"
fontsize1=30 #used for axis labels
fontsize2=30*0.6 #used for tick labels
linewidth1=5 #used for axes
horizontal_line=True #whether to include a  horizontal line to show the average
linewidth2=3 #used for horizontal "average" line
horizontal_line_color="green" 
size_in_inches=(18.5,11) #graph size
                '''
                
            if os.path.exists(path+"para.py"):
                os.remove(path+"para.py")
            f=open(path+"para.py","w")
            f.write(para_file)
            f.close()
            print("Reset parameter file")
            sys.exit()
    
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
    
    return model_path,dataset_path,False,None,no_graph

def create_storage_directory(model_path,dataset_path):
    dir_name="Default/"
    model_name=os.path.splitext(os.path.basename(model_path))[0]
    dataset_name=os.path.splitext(os.path.basename(dataset_path))[0]
    dir_name=model_name+"_"+dataset_name
    
    storage_dir=path+"/storage/"+dir_name+"/"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    
    os.makedirs(storage_dir)
    
    return storage_dir

# TBA: taken from SGDML?
# Assumes that the atoms in each molecule are in the same order.
def read_concat_ext_xyz(f):
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
                z.append(io._z_str_to_z_dict[cols[0]])
            F.append(map(float,cols[4:7]))

    R = np.array(R).reshape(-1,n_atoms,3)
    z = np.array(z)
    E = np.array(E)
    F = np.array(F).reshape(-1,n_atoms,3)

    f.close()
    return (R,z,E,F)
    
def load_dataset(dataset_path):
    
    ext=os.path.splitext(dataset_path)[-1]
    
    #xyz file
    if ext==".xyz":
        try:
            file=open(dataset_path)
            dat=read_concat_ext_xyz(file)
            data={ 'R':np.array(dat[0]),'z':dat[1],'E':np.reshape( dat[2] , (len(dat[2]),1) ),'F':np.array(dat[3]) }
        except:
            return False
    #npz file        
    elif ext==".npz":
        try:
            data=np.load(dataset_path)
        except:
            return False
            
    return data
   
def calculate_errors(model,dataset,cluster_indices):
    print("Calculating errors for each cluster...")
    try:
        gdml=GDMLPredict(model)
    except:
        print("Were not able to read GDML model file.")
        sys.exit(2)
    
    #helping variables
    n_clusters=len(cluster_indices)
    R,F=dataset["R"],dataset["F"]
    mse=[]
    
    #loop through clusters
    #predict results for each cluster
    #calculate error, save in list
    sys.stdout.write( "\r[0/{}] done".format(n_clusters) )
    sys.stdout.flush()
    for i in range(n_clusters):
        cind=cluster_indices[i] #cluster indices
        cr,cf=R[cind],F[cind] #cluster R 
        n_samples,n_atoms,n_dim=cr.shape
        
        #reshaping, necessary for GDML to process it
        cr=np.reshape(cr,(n_samples,n_atoms*n_dim))  
        cf=np.reshape(cf,(n_samples,n_atoms*n_dim))  
        
        #predict forces for all given cluster geometries
        _,cf_pred=gdml.predict(cr)
        err=(cf-cf_pred)**2
        mse.append(err.mean())

        #print out
        sys.stdout.write( "\r[{}/{}] done".format(i+1,n_clusters) )
        sys.stdout.flush()
     
    print("")   
    #order the cluster_indices etc
    sorted_ind=np.argsort(mse)
    mse=np.array(mse)
    cluster_indices=np.array(cluster_indices)
    
    return mse[sorted_ind],cluster_indices[sorted_ind]
    
def error_graph(mse,dir):
    #x axis
    x=np.arange(len(mse))+1
    
    #helping variables
    min,max,avg=np.min(mse),np.max(mse),np.average(mse)
    med=(max+min)/2
    xticks=[]
    yticks=[min,avg,med,max]
    xlabels=[]
    ylabels=["{:.1f}".format(min),"{:.1f}".format(avg),"{:.1f}".format(med),"{:.1f}".format(max)]
    fs=para.fontsize1
    fs2=para.fontsize2
    lw=para.linewidth1
    lw2=para.linewidth2
    
    #create figure
    f,ax1=plt.subplots()
    ax1.bar(x,mse)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels,fontsize=fs)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels,fontsize=fs2)
    plt.xlabel(para.x_axis_label,labelpad=10,fontsize=fs)
    plt.ylabel(para.y_axis_label,fontsize=fs)
    
    for i in ["top","right","bottom","left"]:
        ax1.spines[i].set_visible(False)
    
    ax1.axhline(linewidth=lw,color="black",clip_on=False)
    ax1.axvline(linewidth=lw,color="black")
    if para.horizontal_line:
        ax1.axhline(avg,linewidth=lw2,color=para.horizontal_line_color)   
    f.set_size_inches(para.size_in_inches)
    f.savefig(dir+"graph.png")
    print("Graph saved at {}".format(dir+"graph.png"))
    
    
if __name__=="__main__":
    model_path,dataset_path,graph_only,mse_path,no_graph=parse_arguments(sys.argv[1:])
    
    import para

    #graph-only mode
    if graph_only:
        storage_directory=os.path.dirname(mse_path)+"/"
        try:
            mse=np.loadtxt(mse_path)
        except:
            print("Could not load mse text file at "+mse_path+".")
            sys.exit(2)
        
        error_graph(mse,storage_directory)
        sys.exit()
    
    #try to load the model and dataset
    model,dataset=None,None
    try:
        model=np.load(model_path)
    except:
        print("Could not load model file. Please ensure it is of the right format (.npz)")
        sys.exit(2)
    
    dataset=load_dataset(dataset_path)
    if not dataset:
        print("Could not load dataset file. Please ensure it is of the right format (.npz or .xyz)")
        sys.exit(2)
        
    #create storage directory and create path to it for further functions    
    storage_dir=create_storage_directory(model_path,dataset_path)

    #prepare data
    R=r_to_desc(dataset["R"])
    E=np.array(dataset["E"])
    
    #cluster the data, return indices of each cluster
    cluster_indices=cluster.cluster(R,E,storage_dir)
    
    mse,cluster_indices=calculate_errors(model,dataset,cluster_indices)

    #save cluster indices
    if os.path.exists(storage_dir+"cluster_indices_all.npy"):
        os.remove(storage_dir+"cluster_indices_all.npy")
    np.save(storage_dir+"cluster_indices_all.npy",cluster_indices)

    #save mse as text
    if os.path.exists(storage_dir+"mse_all.txt"):
        os.remove(storage_dir+"mse_all.txt")
    np.savetxt(storage_dir+"mse_all.txt",mse,delimiter=" ")    
    
    #make and store the graph
    if not no_graph:
        error_graph(mse,storage_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
