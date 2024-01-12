#!/bin/bash

########################
#### FlatFlame Right ###
########################

# An effective 1D flame. Burning from left of domain to the right.

# Case details:
#     1atm, 300K, 0.4phi H2 Flame
#     sF = sL
#     lF = lL
#     Markstine number does not matter.

# Domain details:
#       1*lL width
#       16*lL length
#       16Cells across lL


########### run ###########

## compile
export ncores=8;
make -j${ncores} DIM=2 > /dev/null;

## run
mpirun -n 8 ./iamr-levelset2d.gnu.MPI.ex inputs.2d max_step=10 ns.init_dt=1e-6 >> output.txt;
ln -s ../../Tools/probe2d.gnu.ex;
./probe2d.gnu.ex inputs.2d infile= plt00010 vars= density x_velocity axis=0 coord=0 8 > /dev/null;
rm probe2d.gnu.ex;
python compare-results.py;
rm -r output.txt chk????? plt????? plt00010_probe.dat;
