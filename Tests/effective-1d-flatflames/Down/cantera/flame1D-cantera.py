import cantera as ct
import numpy as np
import csv

### Flame Conditions
p = 1. * ct.one_atm  # pressure [Pa]
Tin = 300.           # unburned gas temperature [K]
phi = 0.4            # equiverlance ratio

print("running Cantera ...")
width = 0.1
loglevel = 0 
# Activation Energy Simulations
reactants = {'H2':0.42*phi, 'O2':0.21, 'N2':0.79, 'AR':0.00}  # premixed gas composition
gas = ct.Solution('BurkeDryer_mod_yaml.txt')
gas.TPX = Tin, p, reactants
f = ct.FreeFlame(gas, width=width)
f.max_grid_points=10000
f.set_refine_criteria(ratio=3, slope=0.01, curve=0.01)
f.transport_model = 'mixture-averaged'
f.solve(loglevel=loglevel, auto=True)
print("... success")

print("calculations ...")
sigma = f.density[0] / f.density[-1]
thermodynamic_expansion = f.velocity[-1] - f.velocity[0]
laminar_flamespeed = f.velocity[0]
laminar_thermal_thickness = (f.T[-1] - f.T[0]) /  np.max(np.diff(f.T[:]) / np.diff(f.grid[:]))
print("... success")

with open('ct-flamedata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([sigma])
    writer.writerow([thermodynamic_expansion])
    writer.writerow([laminar_flamespeed])
    writer.writerow([laminar_thermal_thickness])
