#!/bin/bash

cd ../Tools;
make realclean;
cd ../Tests;

echo "Test Left:"
cd effective-1d-flatflames/build;
./cleanup.sh;
cd ../..;
