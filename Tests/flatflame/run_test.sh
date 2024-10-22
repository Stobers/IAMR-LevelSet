#!/bin/bash

export NPROC=16;
export DIM=2;

echo "building tools";
cd ../../Tools/Src;
make -j${NPROC} DIM=${DIM} EBASE=gVolume > /dev/null;
make -j${NPROC} DIM=${DIM} EBASE=probe > /dev/null;
echo "... done";

echo "building IAMR";
cd ../../Tests/flatflame;
make -j${NPROC} DIM=${DIM} > /dev/null;


echo "linking";
ln -s ../../Tools/Src/gVolume${DIM}d.gnu.ex .;
ln -s ../../Tools/Src/probe${DIM}d.gnu.ex .;
ln -s ../../Tools/Scripts/flamespeed.py .;


echo "running IAMR"
rm -r plt*;
mpirun -n ${NPROC} iamr-levelset${DIM}d.gnu.MPI.ex inputs >& output.txt;


echo "processing results"
./gVolume${DIM}d.gnu.ex infiles= plt* > /dev/null;
./probe${DIM}d.gnu.ex inputs infile= plt00100 vars= y_velocity axis=1 coord=8 0;
python flamespeed.py;
