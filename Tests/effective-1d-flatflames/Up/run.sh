#!/bin/bash

########################
#### FlatFlame Down ####
########################

# An effective 1D flame. Burning from top of domain to the bottom.

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
mpirun -n 8 ./iamr-levelset2d.gnu.MPI.ex inputs.2d max_step=200 ns.init_dt=1e-6 >> output.txt;
ln -s ../../../Tools/probe2d.gnu.ex;
./probe2d.gnu.ex inputs.2d infile= plt00200 vars= density y_velocity axis=1 coord=8 0 > /dev/null;
rm probe2d.gnu.ex;

ln -s ../../../Tools/flamespeed2d.gnu.ex;
./flamespeed2d.gnu.ex infiles= plt????? vaxis=0 > /dev/null;
rm flamespeed2d.gnu.ex;

python compare-results.py;
rm -r output.txt chk????? plt????? plt?????_probe.dat flamespeed.dat;
