#!/bin/bash
module load python
conda activate cernroot

pip install energyflow
pip install PyPDF2
pip install tensorflow
pip install omnifold
pip install pandas
# Change this path to your RooUnfold build directory
source /global/homes/r/rmilton/m3246/rmilton/omnifold_paper_plots/RooUnfold/build/setup.sh
