
#cluster parameters
number_of_spacial_clusters=10      #how many clusters to create during the initial agglomerative clustering step
number_of_energy_clusters=5        #how many clusters to separate every cluster of previous step into
initial_spatial_data_points=30000  #determines how many points are used for the initial agglomerative clustering step
                                   #if a memory problem/overflow occurs, or stuck at "Preparing and transforming dataset...", try lowering this
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
    