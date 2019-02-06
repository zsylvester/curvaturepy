# curvaturepy
Code for calculating curvature and migration rate in meandering rivers

Code and data used for the analysis of the relationship between curvature and migration rate, as described in Sylvester et al., (2019), "High curvatures drive river meandering".

Assumes that centerlines for two time steps have been interpreted; the data for these centerlines is stored in the 'data' folder. Each Landsat scene has a corresponding folder under 'data', named after the rivers that are covered. For each scene, there is a 'csv_files' folder that contains the files with the UTM coordinates of the centerlines and banks; the list of indices that define bends and the corresponding 'points of zero migration'.

Key functions are in the 'cline_analysis.py' module. 

The Jupyter notebooks can be used to run the analysis for each river segment and recreate most of the plots in the paper and in the supplementary file.

Computing migration rates requires the 'dp_python' package (https://github.com/dpwe/dp_python). I could only get this to work with Python 2.
