# CosmoCorr  

A Julia package for calculating 2 pt correlation functions for Cosmological analysis. Contains the full naive approach, an implementation of KD-Tree binning, and a novel kmeans method for merging neighboring galaxies at a specified number or granularity. Take's advantage of Julia's multiple dispatch to abstract away the different types of things you may want to correlate. Simply give the positions and quantities of the things you want to correlate using the `corr` function and you will obtain the desired estimate.

To-do, add benchmarks to advise which of the three schemes works best and examine the trade offs between accuracy and speed.

![image][assets/CC_logo.png]
