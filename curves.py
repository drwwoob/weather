"""
Author: Kelly Tao
Date: Nov. 25th
Description: calculate the flow(curves) of data in one single place
"""

import numpy as np
import netCDF4 as nc

# for visualize
import mateplotlib.pyplot as plt

def load_data(name):
    """
    @method load the data into some variable and make it calculatable
    
    @parameters
        name --- string, the path and name of the files to load
        //factors --- [...] list, the factors(columns) we want? or just the whole file, can be put in multiple
        
    @instances
        dataList --- the list of data 
    
    @return void
    """
    ds = nc.Dataset(name)
    
    
def
