from datetime import date
import imageio
from retrying import retry
import urllib.request
import pandas as pd
import numpy as np
import time
import os
import os.path
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy import misc
from io import StringIO
from io import BytesIO
gdal.UseExceptions()
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
'exec(%matplotlib inline)'
# get_ipython().run_line_magic('matplotlib', 'inline')
import urllib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")




# file_name = 'C:/Users/tejav/Jupyter/data/DHS/RWHR61FL.DAT'
# cluster_file = 'C:/Users/tejav/Jupyter/data/DHS/rwanda_clusters_location.csv'
# cluster_all = []
# wealth_all = []
# with open(file_name) as f:
#     for line in f:
#         cluster = int(line[15:23])
#         wealth = int(line[230:238]) / 100000.0
#         cluster_all.append(cluster)
#         wealth_all.append(wealth)

# df = pd.DataFrame({'cluster': cluster_all, 'wlthindf': wealth_all})
# cluster_avg_asset = df.groupby('cluster')['wlthindf'].median().reset_index()
# df_location = pd.read_csv(cluster_file)[['DHSCLUST', 'LATNUM', 'LONGNUM']]
# result = cluster_avg_asset.merge(df_location, how='inner', left_on='cluster', right_on='DHSCLUST')[['cluster', 'wlthindf', 'LATNUM', 'LONGNUM']]
# result.rename(columns={'LATNUM': 'latitude', 'LONGNUM':'longitude'}, inplace=True)
# result.to_csv('C:/Users/tejav/Jupyter/intermediate_files/rwanda_cluster_avg_asset_2010.csv', index=False)


########################################### cell 1 ###############################################



def read_raster(raster_file):

    raster_dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    # get project coordination
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    # Loop through all raster bands
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
        no_data_value = band.GetNoDataValue()
    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    # Get the metadata of the raster
    geo_transform = raster_dataset.GetGeoTransform()
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = geo_transform
    
    # Get location of each pixel
    x_size = 1.0 / int(round(1 / float(x_size)))
    y_size = - x_size
    y_index = np.arange(bands_data.shape[0])
    x_index = np.arange(bands_data.shape[1])
    top_left_x_coords = upper_left_x + x_index * x_size
    top_left_y_coords = upper_left_y + y_index * y_size
    # Add half of the cell size to get the centroid of the cell
    centroid_x_coords = top_left_x_coords + (x_size / 2)
    centroid_y_coords = top_left_y_coords + (y_size / 2)

    return (x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data)


# Helper function to get the pixel index of the point
def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx



########################################### cell 2 ###############################################

raster_file = 'C:/Users/tejav/Jupyter/data/nighttime_image/F182010.v4d_web.stable_lights.avg_vis.tif'
x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data = read_raster(raster_file)

np.savez('C:/Users/tejav/Jupyter/intermediate_files/nightlight.npz', top_left_x_coords=top_left_x_coords, top_left_y_coords=top_left_y_coords, bands_data=bands_data)


########################################### cell 3 ###############################################



def get_nightlight_feature(sample):
    idx, wealth, x, y = sample
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    # Select the 10 * 10 pixels
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    luminosity_100 = []
    for i in range(left_idx, right_idx + 1):
        for j in range(up_idx, low_idx + 1):
            # Get the luminosity of this pixel
            luminosity = bands_data[j, i, 0]
            luminosity_100.append(luminosity)
    luminosity_100 = np.asarray(luminosity_100)
    max_ = np.max(luminosity_100)
    min_ = np.min(luminosity_100)
    mean_ = np.mean(luminosity_100)
    median_ = np.median(luminosity_100)
    std_ = np.std(luminosity_100)
    return pd.Series({'id': idx, 'max_': max_, 'min_': min_, 'mean_': mean_, 
                      'median_': median_, 'std_': std_, 'wealth': wealth})


clusters = pd.read_csv('C:/Users/tejav/Jupyter/intermediate_files/rwanda_cluster_avg_asset_2010.csv')
data_all = clusters.apply(lambda x: get_nightlight_feature([x['cluster'], x['wlthindf'], x['longitude'], x['latitude']]), axis=1)
data_all.to_csv('C:/Users/tejav/Jupyter/intermediate_files/DHS_nightlights.csv', index=None)




########################################### cell 4 ###############################################

# ax = sns.regplot(x="mean_", y="wealth", data=data_all)
# plt.xlabel('Average nighttime luminosity')
# plt.ylabel('Average cluster wealth')
# plt.xlim([0, 50])





########################################### cell 5 ###############################################


def get_shp_extent(shp_file):
    print("shp_file_ext")
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(shp_file, 0)
    inLayer = inDataSource.GetLayer()
    extent = inLayer.GetExtent()
    # x_min_shp, x_max_shp, y_min_shp, y_max_shp = extent
    return extent





########################################### cell 6 ###############################################


#@retry(wait_exponential_multiplier=1000, wait_exponential_max=3600000)
def save_img(url, file_path, file_name):  
    # a = urllib.request.urlopen(url).read()
    # print("a")
    # b = BytesIO(a)
    # print("b")
    image = plt.imread(url)
    print("image")

    if np.array_equal(image[:,:10,:],image[:,10:20,:]):
        print("=======")
        pass
    else:
        print('---')
        plt.imsave(file_path + file_name, image[50:450, :, :])

# reading shapefile
inShapefile = "C:/Users/tejav/Jupyter/data/shp/Sector_Boundary_2012/Sector_Boundary_2012.shp"
x_min_shp, x_max_shp, y_min_shp, y_max_shp = get_shp_extent(inShapefile)

left_idx, top_idx = get_cell_idx(x_min_shp, y_max_shp, top_left_x_coords, top_left_y_coords)
right_idx, bottom_idx = get_cell_idx(x_max_shp, y_min_shp, top_left_x_coords, top_left_y_coords)

key_file = open("C:/Users/tejav/Jupyter/data/key.txt","r")

key = key_file.readline()
m = 1


today_date = date.today()
date_format = today_date.strftime("%d%m%Y")

log_file=open("log.txt_" + date_format,"a")

log_file.write(str(left_idx) + " " + str(right_idx) + "\n")
log_file.write(str(top_idx) + " " + str(bottom_idx))

log_file.close()

print(left_idx, right_idx)
print(top_idx, bottom_idx)

for i in range(25280, right_idx + 1):
    log_file=open("log.txt_" + date_format,"a")
    for j in range(9219, bottom_idx + 1):
        lon = centroid_x_coords[i]
        lat = centroid_y_coords[j]
        print(lon,lat)
        url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + str(lon) + '&zoom=16&size=400x400&maptype=satellite&key=' + key
        lightness = bands_data[j, i, 0]
        file_path = 'C:/Users/tejav/Jupyter/google_image/' + str(lightness) + '/'
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file_name = str(i) + '_' + str(j) +'.jpg'
        print(url, "--", file_name, file_path)
        save_img(url, file_path, file_name)
        log_str= str(i) +" "+ str(j) +"\n"+url+"\n"+ file_name +","+ file_path + "\n"
        log_file.write(log_str)
        log_file.write("--------------------------------\n")
        print(file_path, file_name)
        if m % 100 == 0:
            print(m)
        m += 1
