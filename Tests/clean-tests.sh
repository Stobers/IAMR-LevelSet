#!/bin/bash

cd Tools;
make realclean;
cd ..;

echo "Test Left:"
cd effective-1d-flatflames/Left;
./cleanup.sh;
cd ../..;

echo "Test Right:"
cd effective-1d-flatflames/Right;
./cleanup.sh;
cd ../..;

echo "Test Up:"
cd effective-1d-flatflames/Up;
./cleanup.sh;
cd ../..;

echo "Test Down:"
cd effective-1d-flatflames/Left;
./cleanup.sh;
cd ../..;
