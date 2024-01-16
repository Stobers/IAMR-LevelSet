#!/bin/bash


echo "#### Building Tools ####"
cd ../Tools;
make -j8 DIM=2 EBASE=probe > /dev/null;
make -j8 DIM=2 EBASE=flamespeed > /dev/null;
cd ../Tests;

echo "#### Effective 1D Flat Flames ####"
cd effective-1d-flatflames/build;

echo "Test Up:"
cp Up/* .;
./run.sh;
rm prob_init.cpp inputs.2d compare-results.py run.sh;

echo "Test Down:"
cp Down/* .;
./run.sh;
rm prob_init.cpp inputs.2d compare-results.py run.sh;

echo "Test Left:"
cp Left/* .;
./run.sh;
rm prob_init.cpp inputs.2d compare-results.py run.sh;

echo "Test Right:"
cp Right/* .;
./run.sh;
rm prob_init.cpp inputs.2d compare-results.py run.sh;


## Going Home
cd ../..;
