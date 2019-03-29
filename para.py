
#cluster parameters
number_of_spacial_clusters=10
number_of_energy_clusters=5
initial_spatial_data_points=5000

#graph parameters
x_axis_label="Cluster number1"
y_axis_label="Mean squared average"
fontsize1=30 #used for axis labels
fontsize2=30*0.6 #used for tick labels
linewidth1=5 #used for axes
horizontal_line=True #if True, includes a horizontal line to show the average error 
linewidth2=3 #used for horizontal "average error" line
horizontal_line_color="green" 
size_in_inches=(18.5,11) #graph size
include_population=True  #indicates the cluster population for every cluster 
total_population=False  #if include_population is True, the population will instead be shown as a total population curve
order_by_energy=False     #if true, the graph will order the clusters by their average energy rather than error
reverse_order=False  #if False, orders from lowest error/energy to highest (left to right), reversed otherwise