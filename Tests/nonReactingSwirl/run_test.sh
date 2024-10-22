#!/bin/bash

export NPROC=16;
export DIM=2;

echo "building tools";
cd ../../Tools/Src;
make -j${NPROC} DIM=${DIM} EBASE=gVolume > /dev/null;
echo "... done";

echo "building IAMR";
cd ../../Tests/nonReactingSwirl;
make -j${NPROC} DIM=${DIM} > /dev/null;


echo "linking";
ln -s ../../Tools/Src/gVolume${DIM}d.gnu.ex .;
ln -s ../../Tools/Scripts/dvdt.py .;


echo "running IAMR"
rm -r plt*_exact plt*_reinit;
mpirun -n ${NPROC} iamr-levelset${DIM}d.gnu.MPI.ex inputs.exact >& output_exact.txt;
for i in plt?????;
do
    mv ${i} ${i}_exact;
done
mpirun -n ${NPROC} iamr-levelset${DIM}d.gnu.MPI.ex inputs.reinit >& output_reinit.txt;
for i in plt?????;
do
    mv ${i} ${i}_reinit;
done


echo "processing results"
./gVolume2d.gnu.ex infiles= plt*_exact > /dev/null;
mv gVolume.dat gVolume_exact.dat;
./gVolume2d.gnu.ex infiles= plt*_reinit > /dev/null;
mv gVolume.dat gVolume_reinit.dat;
