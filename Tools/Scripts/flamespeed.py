import numpy as np


data = np.loadtxt("gVolume.dat");
t = data[:,0]
v = data[:,1]
dvdt = (v[-1] - v[1]) / (t[-1] - t[1])
print("flamespeed = "+str(dvdt))
