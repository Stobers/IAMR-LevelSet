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

export ncores=8;

## compile
make realclean;
make -j${ncores} DIM=2;
make -j${ncores} DIM=3;

## run 2d
mpirun -n ${ncores} ./iamr-levelset2d.gnu.MPI.ex inputs.2d max_step=10 ns.init_dt=1e-6
for i in plt?????;
do
    mv ${i} ${i}.2d;
done
mv plt?????.2d results/.
## run 3d
mpirun -n ${ncores} ./iamr-levelset3d.gnu.MPI.ex inputs.2d max_step=10 ns.init_dt=1e-6
for i in plt?????;
do
    mv ${i} ${i}.3d;
done
mv plt?????.3d results/.
