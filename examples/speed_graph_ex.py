from lucass_tools.speed_graph import speedGraph
import numpy as np

#Parameters
alpha = 1
K = 50  
rho = 1 
b1 = 1 
b2 = 1 
b3 = 1 
b4 = 1 
b5 = 1 
b6 = 0 
#initial conditions
X0 = np.array([1,0,0,0,0,0])

def bio_model(X,t=0):
    S1=X[0]
    S2=X[1]
    S3=X[2]
    S4=X[3]
    S5=X[4]
    S6=X[5]
    return np.array([ K/(1+alpha*(S6**rho)) - b1*S1,
     b1*S1 - b2*S2,
     b2*S2 - b3*S3,
     b3*S3 - b4*S4,
     b4*S4 - b5*S5,
     b5*S5 - b6*S6])

linMod = speedGraph(bio_model,
                    X0,
                    var_labels= ['s1','s2','s3','s4','s5','s6'],
                    vars_to_plot= ['s1','s6'],
                    tInit=0,
                    tFinal=100,
                    tSpace=3000,
)
print('Finished.')
linMod.plotGraphs(with_eqs=False)