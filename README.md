# CosmoCorr

A Julia package for calculating 2 pt correlation functions for Cosmological analysis. Contains the full naive approach, an implementation of KD-Tree binning, and a novel heirarchical clustering method for merging neighboring clusters at a specified number or granularity. Take's advantage of Julia's multiple dispatch to abstract aways the different types of things you may want to correlate. Simply give the positions and quantities of the things you want to correlate using the `corr` function and you will obtain the desired estimate.
