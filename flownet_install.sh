#!/bin/bash
cd ./flownet/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ..
