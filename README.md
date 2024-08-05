# unfolding
This repository contains notebooks for binned unfolding using Iterative Bayesian Unfolding + a decision-tree based Omnifold, as well as unbinned unfolding using decision-tree based Omnifold. 
## Prerequisites
To run all these notebooks, the following are required:
- RooUnfold
- Numpy
- Matplotlib.pylot
- sklearn
- mplhep

## IBU
The IBU notebook has two components:
- Implementing IBU by hand and comparing it to the output from RooUnfold
- Writing Omnifold variables in terms of IBU variables and comparing Omnifold with IBU
Most of the functions are contained in `unfolding.py`. There are helper functions, mainly for histogram manipulation and data generation, in `data_handler.py`.

## Decision tree Omnifold
An Omnifold implementation using a decision tree classifier is in `omnifold_decisiontrees.ipynb`, and it directly compares the unfolded results to RooUnfold IBU. This notebook currently only does binned unfolding, with unbinned unfolding currently in-progress. This notebook needs sklearn to use the decision trees.
