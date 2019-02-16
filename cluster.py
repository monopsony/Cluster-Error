import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import os as os
import sys as sys

path = os.path.dirname(os.path.realpath(__file__))


#returns the cluster number that the sample belongs to (shortest distance to center)
def find_smallest_max_distance_index(sample,clusters):
    g=np.zeros( len(clusters))
    for i in range(len(clusters)):
        g[i]=np.max(np.sum(np.square(clusters[i]-sample),1))   #numpy difference=>clusters[c]-sample elementwise for each c
    return np.argmin(g)
    
def cluster(R,E,dir):
    import para

    print("Starting clusterisation")
    #making sure they're numpy arrays
    R,E=np.array(R),np.array(E)
    
    if R.shape[0]<initial_spatial_data_points:
        print("Not enough points in dataset. Lower the initial_spatial_data_points parameter in the para.py file or add more points to the database.")
        return
        #TBA error handling
        
    #helping variables    
    print("Preparing and transforming dataset...")
    n1=para.number_of_spacial_clusters
    n2=para.number_of_energy_clusters
    ni=para.initial_spatial_data_points
    cluster_ind,clusterE,clusterR=[],[],[]
    ind_all=np.arange(E.shape[0])
    ind_init=np.random.choice(ind_all,ni)   
    ind_rest=np.delete(ind_all,ind_init)
    R_init=R[ind_init]
    R_rest=R[ind_rest]


    #perform agglomerative clustering
    M=np.square(euclidean_distances(R_init,R_init))
    print("Starting Agglomerative clustering...")
    clusterLabels=AgglomerativeClustering(affinity="precomputed",n_clusters=n1,linkage='complete').fit_predict(M)
    print("Agglomerative clustering done")
    
    for i in range(n1):
        ind=np.concatenate(np.argwhere(clusterLabels==i))

        #convert back to initial set of indices (since these are indices from a subset of the entire dataset)
        ind=ind_init[ind]

        cluster_ind.append(ind.tolist())
        clusterE.append(np.array(E[cluster_ind[i]]))
        clusterR.append(np.array(R[cluster_ind[i]]))
    
    #divide rest into clusters
    #using smallest max distance as criterion
    #+ni to find the old index back from entire dataset
    print("Clustering rest of data...")
    outs=np.trunc(np.linspace(0,len(R_rest),99))
    for i in range(len(R_rest)):
        
        #%done output
        if i==outs[0]:
            if i==0:
                ch=0
            else:
                ch=float(i)/len(R_rest)*100
            sys.stdout.write("\r[{:.0f}%] done".format(ch+1))
            sys.stdout.flush()
            outs=np.delete(outs,0)
    
        rc=np.array(R_rest[i])
        c=find_smallest_max_distance_index(rc,clusterR) #c is the cluster# it belongs to
        cluster_ind[c].append(ind_rest[i])
    
    print("")
    
    #now perform energy clusters
    #for every single spacial cluster
    print("Reclustering by energy...")
    clusterLabels2=[]
    for i in range(n1):
        sys.stdout.write( "\r[{}/{}] done".format(i+1,n1) )
        sys.stdout.flush()
        c=E[cluster_ind[i]]
        labels=MiniBatchKMeans(n_clusters=n2,init="k-means++").fit_predict(c)
        clusterLabels2.append(labels)
        
    #redefine each mini cluster uniquely
    #fill in cluster data into n1*n2 separate clusters
    cluster_ind2=[[]]*(n1*n2) 
    for i in range(n1):
        labels=clusterLabels2[i]
        indices=np.array(cluster_ind[i])
        
        for j in range(n2):
            ind=np.concatenate(np.argwhere(labels==j).tolist())
            cluster_ind2[i*n2+j]=indices[ind]
           
    
    print("")
    
    return cluster_ind2
    