# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: mv-mcmc.py                                        #
#    DATE: 28 FEB 2025                                       #
# ********************************************************** #

from scipy import io
import numpy as np
import pandas as pd
import random
import geotools as gt

# Import Martha's Vineyard boundary file
mv_shoreline = gt.read_kml_points('../data/mv_shorline.kml')

# Import tidal current data (.mat -> dict)
data = io.loadmat('../data/tidal_data_2022_07_30.mat')
print(data.keys())
u = data['U_store']
v = data['V_store']
t_local = data['glocals']
t = data['thours']
lat = data['lat']
lon = data['lon']
waypts = data['waypts']



test_points = [(-70.65225869587395,41.5120999095306), (-70.64950755788055,41.45247996887631), (-70.63737497239445,41.47278559961993), (-70.62405760185246,41.46051049539149), (-70.62022583586396,41.48223373922173), (-70.60650952595303,41.47120735275431), (-70.60510216769521,41.48801204362773), (-70.59803325641673,41.47963889617993)]

for i in range(len(test_points)):
    print(gt.is_point_in_region(test_points[i], mv_shoreline))
