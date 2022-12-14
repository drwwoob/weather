"""
Author: Kelly Tao
Date: Nov. 25th
Description: calculate the kmean score for the given NetCDF4 data with n-clusters 3-15
"""

import os
import numpy as np
import netCDF4 as nc

# for analyze
#from sklearn.svm import SVC
from sklearn.cluster import KMeans

# for visualize
import matplotlib.pyplot as plt

def load_data(name):
    """
    @method
        load the data into some variable and make it calculatable
    
    @parameters
        name --- string, the path and name of the files to load
        
#    @instances
#        dataList --- the list of data
    
    @return
        ds --- the nCDF file of the given name
    """
    dir = os.path.dirname(__file__)
    f = os.path.join(dir, 'data', name)
    ds = nc.Dataset(f)
    return ds
    

    
def to_array(ds, factor, starting_time, lasting_time, method):
    """
    @method
        calculate the change in data and make a list of it
        
    @parameters
        ds --- the data file
        factor --- the name of the factor we are calculating
        starting_time --- the starting time to read the data
        lasting_time --- to read how much data
        
    @return:
        data --- a 3d array for the given data, dimention as [lattitude, longtitude, the list of data]
    """
    
    size = ds[factor][:, :, :].shape
        
    data = np.zeros((size[1], size[2], len(method(ds[factor][starting_time: starting_time + lasting_time, 0, 0]))))
    for latt in range(size[1] - 1):
        for long in range(size[2] - 1):
            data[latt][long] = method(ds[factor][starting_time: starting_time + lasting_time, latt, long])
            
    return data
    
def deviation(area_data):
    """
    @method
        calculate the change in data through time in a certain area(given)
        
    @parameters
        area_data --- a list of number
    
    @return
        area_deviation --- a list of number
    """
    area_deviation = np.zeros(len(area_data) - 1)
    for i in range(len(area_data) - 2):
        if area_data[i] == np.nan or area_data[i + 1] == np.nan:
            area_deviation[i] = 0
        else:
            area_deviation[i] = area_data[i + 1] - area_data[i]
    
    return area_deviation
    
def plot(data_list):
    """
    @method
        plot the given list
    @return
        void
    """
    
    xpoints = list(range(0,len(data_list)))
    plt.plot(xpoints, data_list)
    plt.show()
    
    
def sort_in_years(dataset):
    """
    @method
        sort the given data by seperating them in years
        
    @parameters
        dataset --- the 3d list of data in the format [lattitude, longtitude, all data]
        
    @return
        year_list --- the 4d list including years, as [year, lattitude, longtitude, data per month]
    """
    size = dataset.shape
    num_years = int(size[2] / 12)
    if size[2] % 12 > 0:
        num_years += 1
    
    year_list = np.zeros((num_years, size[0], size[1], 12))
    for latt in range(size[0]):
        for long in range(size[1]):
            for year in range(num_years):
                if size[2] >= (year + 1) * 12:
                    year_list[year][latt][long] = dataset[latt][long][year * 12 : (year + 1) * 12]
                else:
                    year_list[year][latt][long] = np.append(dataset[latt][long][year * 12 : (year + 1) * 12], np.zeros(12 - len(dataset[latt][long][year * 12 : (year + 1) * 12])))
                    
    return year_list
    
    
def sort_in_coord(dataset_in_year):
    """
    @return
        data_with_coord --- the 2d list in the format [[lattitude, longtitude, data in a year], [same latitude, same longtitude, data in the next year], ...]
    """
    size = dataset_in_year.shape
    data_with_coord = np.zeros((size[0] * size[1] * size[2], size[3] + 2))
    i = 0
    for lat in range(size[1]):
        for long in range(size[2]):
            for year in range(size[0]):
                coord = [lat, long]
                coord.extend(dataset_in_year[year][lat][long][:])
                data_with_coord[i] = coord
                i += 1
    return data_with_coord
    

def sort_with_annual(dataset_in_year):
    """
    @return
        dataset_in_year --- the modified list of data sorted in year with the last data in the list the annual precip
    """
    
    size = dataset_in_year.shape
    for lat in range(size[1]):
        for long in range(size[2]):
            for year in range(size[0]):
                np.append(dataset_in_year[year][lat][long], (sum(dataset_in_year[year][lat][long])))
    return dataset_in_year


def k_means_apply(ds, n_clusters):
    """
    @method
        fit the data with kMeans
        
    @return
        the kmean model
    """
    size = ds.shape
    X = ds.reshape(size[0] * size[1], -1)
    kmeans = KMeans(n_clusters = n_clusters, max_iter = 100)
    kmeans.fit(X)
    return kmeans
    
def find_close(diviations):
    """
    @method
        find what's similar to the current data
    
    @return
        similarity --- a 3d array listing the given data, with the dimentions [lattitude, longtitude, the list of similarity of other data with itself]
    """
   
   
   
# below are methods for testing
    
def print_data(ds):
    """
    @method
        print the data on the terminal, for test purpose
        
    @parameters
        ds --- the file we are seeing
        
    @return
        void
    """
    print(ds)
#    print(ds.__dict__)
#    for var in ds.variables.values():
#        print(var)
    time = ds['time'][:]
    print(len(time))
    print(time)
    lat = ds['lat'][:]
    print(lat)
    lon = ds['lon'][:]
    print(lon)
    print(ds['air'][:, :, :])
#    prcp = ds['precip'][0,0,0]
#    prcp = ds['precip'][:, :, :]
#    for i in range(prcp.shape[0]):
#    for i in range(20):
#    for latt in range(prcp.shape[1]):
#        print(latt)
#        print(ds['precip'][0, latt, :])
#    print(prcp)

def small_to_array(ds, factor, starting_time, lasting_time, method):
    """
    @method
        diviation is too big and takes too long to run so here i will just grab some small data to test
                
    @parameters
        ds --- the data file
        factor --- the name of the factor we are calculating
        starting_time --- the starting time to read the data
        lasting_time --- to read how much data
        
    @instances
        lat, lon --- the coordinates of some chosen places
    
    @return:
        data --- a 3d array for the given data, dimention as [lattitude, longtitude, data]
    """
    
    
    #places:
    lat = [ds['lat'][35:40]]
    lon = [ds['lon'][50:55]]
    
    data = np.zeros((5, 5, len(method(ds[factor][starting_time: starting_time + lasting_time, 0, 0]))))
    for latt in range(4):
        for long in range(4):
            data[latt][long] = method(ds[factor][starting_time: starting_time + lasting_time, latt, long])
            
    return data
    
    
def average_all(year_div):
    """
    @return
        a list of the average of the given data, in the format of [lattitude, longtitude, avarage data]
    """
    average_with_year = np.zeros((year_div.shape[1], year_div.shape[2], 12))
    
    for latt in range(year_div.shape[1]):
        for long in range(year_div.shape[2]):
            year_trim = []
            for year in range(year_div.shape[0]):
                year_trim.append(year_div[year][latt][long][:])
            year_trim = np.array(year_trim).reshape(-1, 12)
            average_with_year[latt][long] = average(year_trim)
    return average_with_year
    
    
def average(year_div_trimmed):
    """
    @method
        find the mean value with the last two dementions of year_div
        
    @return
        average --- the average of the given list
    """
    average = np.zeros(12)
    
    for month in range(12):
        month_calc = []
        for year in range(year_div_trimmed.shape[0]):
            if year_div_trimmed[year][month] != 0:
                month_calc.append(year_div_trimmed[year][month])
        if month_calc != []:
            average[month] = np.mean(month_calc) / len(month_calc)
        else:
            average[month] = 0
    return average
    
    
def get_curve(dataset, degree):
    """
    @method
        get the curves of the given datas
    
    @parameters
        dataset --- the given 2d data in the format [lattitude, longtitude, datalist]
        degree --- an integer of the polynomial
        
    @return
        curves --- the same size array as [lattitude, longtitude, degree]
    """
    size = dataset.shape
    curves = np.zeros((size[0], size[1], degree + 1))
    for lat in range(size[0]):
        for long in range(size[1]):
            curves[lat][long] = get_single_curve(dataset[lat][long][:], degree)
    return curves

    
def get_single_curve(data, degree):
    """
    @method
        get the curve of the given data
    """
    if(data.all() == 0):
        return np.zeros(degree + 1)
    y = list(range(0, len(data - 1)))
    poly = np.polyfit(data, y, deg = degree)
    return poly
    
    
def plot_scatter(model, data):
    groups = [[] for i in range(len(model.cluster_centers_))]
    i = 0
    for lat in range(0, data.shape[0]):
        for long in range(0, data.shape[1]):
            groups[model.labels_[i]].append([lat, long])
            i += 1
            
    for group in groups:
        x = [coord[0] for coord in group]
        y = [coord[1] for coord in group]
        plt.scatter(x, y, alpha=0.3)
    plt.show()
#        print(coord[:][0])
#        print(len(coord))
#        for i in range(len(coord)):
        

def plot_scatter_coord(model, data):
    groups = [[] for i in range(len(model.cluster_centers_))]
    for i in range(len(data)):
        groups[model.labels_[i]].append([data[i][0:2]])
        
    for group in groups:
        x = [coord[0][0] for coord in group]
        y = [coord[0][1] for coord in group]
        plt.scatter(x, y, alpha=0.04)
    plt.show()
    
    
def predict(model, div):
    """
    @method
        get the score of the testing data
    
    @return
        the score
    """
        
    size = div.shape
    return model.score(div.reshape(size[0] * size[1], -1))
    
    
def show_score(ds, test_ds, have_plot, with_coord, k_start, k_end):
    """
    @method
        fit the data into the k-mean model and calculate the score for each n_cluster
    """

    score_list = []
    for i in range(k_start, k_end):
        model = k_means_apply(ds, i)
        score = predict(model, test_ds)
        score_list.append(score)
        if (have_plot):
            if (with_coord):
                plot_scatter_coord(model, ds)
            else:
                plot_scatter(model, ds)
        
    return score_list
    
def main():
    ds_precip = load_data("precip.mon.mean.nc") # 1991 - 2021
#    ds_temp = load_data("air.mon.anom.median.nc") # 1850 - 2021
#    ds_temp = load_data("air.2x2.250.mon.anom.comb.nc") # 1880 - 2022
#    temp_starting_time = (1991 - 1880) * 12
    train_lasting_time = (2015 - 1991) * 12 # the number of month for the training data
#    div = small_to_array(ds_precip, 'precip', 0, train_lasting_time, deviation)
    test_lasting_time = (2020 - 2015) * 12
            
#
#    for pure average per year from diviation
    div_precip = average_all(sort_with_annual(sort_in_years(to_array(ds_precip, 'precip', 0, train_lasting_time, deviation))))
    test_precip_div = average_all(sort_with_annual(sort_in_years(to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, deviation))))
    score_list = show_score(div_precip, test_precip_div, True, False, 3, 15)
#
#


    
#   --- for testing purposes

#    div_temp = average_all(sort_with_annual(sort_in_years(to_array(ds_temp, 'air', temp_starting_time, train_lasting_time, deviation))))
#    test_temp_div = average_all(sort_with_annual(sort_in_years(to_array(ds_temp, 'air', temp_starting_time + train_lasting_time, test_lasting_time, deviation))))
#    score_list = show_score(div_temp, test_temp_div, True, False, 3, 15)
#
#    #for pure poly score 2
#    poly = get_curve(average_all(sort_in_years(to_array(ds_precip, 'precip', 0, train_lasting_time, deviation))), 2)
#    poly_test = get_curve(average_all(sort_with_annual(sort_in_years(to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, deviation)))), 2)
#    score_list = show_score(poly, poly_test, True, False, 3, 15)
    
    
#    #for pure poly score 3
#    poly = get_curve(average_all(sort_in_years(to_array(ds_precip, 'precip', 0, train_lasting_time, deviation))), 3)
#    poly_test = get_curve(average_all(sort_with_annual(sort_in_years(to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, deviation)))), 3)
#    score_list = show_score(poly, poly_test, True, False, 3, 15)

    #for all data raw
#    raw = to_array(ds_precip, 'precip', 0, train_lasting_time, lambda a : a)
#    test_precip_raw = to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, lambda a : a)
#    print(raw.size)
#    print(test_precip_raw.size)
    
#    for average data raw
#    raw = average_all(sort_in_years(to_array(ds_precip, 'precip', 0, train_lasting_time, lambda a : a)))
#    test_precip_raw = average_all(sort_in_years(to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, lambda a : a)))
#    score_list = show_score(raw, test_precip_raw, True, False, 3, 4)
    
    #for all data with coord
#    coord = sort_in_coord(sort_in_years(to_array(ds_precip, 'precip', 0, train_lasting_time, lambda a : a)))
#    test_coord = sort_in_coord(sort_in_years(to_array(ds_precip, 'precip', 0 + train_lasting_time, test_lasting_time, lambda a : a)))
#    score_list = show_score(coord, test_coord, True, True, 3, 10)

#    print(deviation(ds_precip, 'precip'))
#    print_data(ds_temp)
#    print_data(ds_precip)
#    div = small_diviation(ds_temp, 'air', trim_temp, train_lasting_time)
#    print(div)

#    print(div.shape)
#    plot(div[1][1])
#    plot(year_div[0][1][1])
#    plot(year_div[1][1][1])
#    plot(year_div[2][1][1])
#    plot(average(year_div[:][1][1]))

#    print(year_div)
#    print(average_all(year_div))
    
#    plot(div[2][3])
#    plot(div[4][4])

    plot(score_list)
    
    
if __name__ == "__main__":
    main()
