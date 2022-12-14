# rain!

To run this code you need to first active the sklearn environment

also, since the format of data are netCDF4, you need to install it with    pip install netCDF4

Create a file on the root with the nme "data", and put the following data in

The data are from the following: 
https://www.psl.noaa.gov/data/gridded/data.cmap.html
at the bottom left, click on the "download files" on the second row with "percipitation, mean, surface, monthly"
Then, in the file listed, click on the file "precip.mon.mean.nc" to download.


After the setup, you can now run the file "curves.py" in the root to see the results.


The result of putting in coordinates

<img width="565" alt="image" src="https://user-images.githubusercontent.com/46990486/206928547-c7dcd616-48d0-4389-98e8-005322798dd4.png">
And the score is presented as

<img width="608" alt="image" src="https://user-images.githubusercontent.com/46990486/206928577-d7a08809-0976-465c-80f2-c5c6d54c757d.png">

Though the score is comparatively better, this does not seems like the right direction
