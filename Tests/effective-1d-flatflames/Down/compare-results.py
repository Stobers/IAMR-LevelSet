import numpy as np

ct_data    = np.loadtxt("ct-flamedata.csv")
ls_data    = np.loadtxt("plt00200_probe.dat")
ls_sL_data = np.loadtxt("flamespeed.dat")

ct_sigma  = ct_data[0]
ct_ufluid = ct_data[1] # fluid acceleration due to thermodynamic expansion
ct_sL     = ct_data[2]

ls_sigma  =  ls_data[0,1] / ls_data[-1,1]
ls_ufluid =  ls_data[-1,2] - ls_data[0,2] # fluid acceleration due to thermodynamic expansion
ls_sL     =  np.mean(ls_sL_data[1:-1-1,1])

error_sigma = np.abs(((ls_sigma - ct_sigma) / ct_sigma) * 100.)
error_ufluid = np.abs(((ls_ufluid - ct_ufluid) / ct_ufluid) * 100.)
error_flamespeed = np.abs(((ls_sL - ct_sL) / ct_sL) * 100.)

if (error_sigma < 2.):
    print("--- pass, sigma has error "+str(round(error_sigma,2))+"%")
else:
    print("--- fail, sigma has error "+str(round(error_sigma,2))+"%")
if (error_ufluid < 2.):
    print("--- pass, fluid velocity from expansion has error "+str(round(error_ufluid,2))+"%")
else:
    print("--- fail, fluid velocity from expansion has error "+str(round(error_ufluid,2))+"%")
if (error_flamespeed < 5.):
    print("--- pass, global flamespeed has error "+str(round(error_flamespeed,2))+"%")
else:
    print("--- fail, global flamespeed has error "+str(round(error_flamespeed,2))+"%")
