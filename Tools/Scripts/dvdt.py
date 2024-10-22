import numpy as np


data = np.loadtxt("gVolume_exact.dat");
t_exact = data[:,0]
v_exact = data[:,1]
dv_exact = 100.*(v_exact[-1] - v_exact[0]) / v_exact[0]
print("exact volume loss = "+str(dv_exact)+"%")
del data

data = np.loadtxt("gVolume_reinit.dat");
t_reinit = data[:,0]
v_reinit = data[:,1]
dv_reinit = 100.*(v_reinit[-1] - v_reinit[0]) / v_reinit[0]
print("reinit volume loss = "+str(dv_reinit)+"%")
del data
