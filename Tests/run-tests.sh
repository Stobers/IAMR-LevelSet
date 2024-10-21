#!/bin/bash

export NPROC=16;

echo "#### Building Tools ####"
cd ../Tools;
make -j${NPROC} DIM=2 EBASE=probe;
make -j${NPROC} DIM=2 EBASE=flamespeed;
cd ../Tests;

echo "#### Effective 1D Flat Flames ####"
cd flatflame;
make -j${NPROC};
mpirun -n ${NPROC} iamr-levelset2d.gnu.MPI.ex inputs;

ln -s ../../Tools/*.ex .

./probe2d.gnu.ex inputs infile= plt00100 vars= density y_velocity axis=1 coord=8 0;
./flamespeed2d.gnu.ex infiles= plt????? vaxis=0;
cat flamespeed.dat;
cat plt00100_probe.dat;
###python compare-results.py;
###rm -r output.txt chk????? plt????? plt?????_probe.dat flamespeed.dat;

## Going Home
cd ..;
