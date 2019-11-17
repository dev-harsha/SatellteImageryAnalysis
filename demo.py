#!/usr/bin/env python
# coding: utf-8

# # Poverty Prediction by Satellite Imagery
# 
# *Is it feasible for you to estimate the standard of living or energy consumption of a site based night-time satellite imagery?This means whether we can do the satellite imagery processing for socioeconomic analysis?*
# 
# For this project, we will focus on the country of Rwanda. We will need to download three distinct datasets, including DHS data, satellite data from the Google Maps API, as well as nighttime luminosity data. The DHS data requires registration (which can take several days to be approved), and the Google Maps API is rate-limited, so it will be slow. The deep learning section may also take several hours to compute
# 
# ## Overview
# 
# These are the key steps in the problem set:
# 
# 1. Download satellite night lights images from NOAA
# 2. Download DHS data for Rwanda
# 3. Test whether night lights data can predict wealth, as observed in DHS
# 4. Download daytime satellite imagery from Google Maps
# 5. Test whether basic features of daytime imagery can predict wealth
# 6. Extract features from daytime imagery using deep learning libraries
# 7. Replicate final model and results of Jean et al (2016)
# 8. Construct maps showing the predicted distribution of wealth in Rwanda
# 

# --------------

# # 1. Download nightlights for Rwanda
# 
# - **INPUT**:
#  - None
# - **OUTPUT**: 
#  - `F182010.v4d_web.stable_lights.avg_vis.tif`: Single image file giving nightlights intensity around the world
# 
# Go to the [DMSP-OLS website](https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html) and download the satellite nighttime luminosity data (roughly 400MB). We will use the one from 2010. The archive they provide constains several files. Feel free to explore these files. We will only be using the file F182010.v4d_web.stable_lights.avg_vis.tif.
# 

# In[2]:


# import wget

# night_image_url = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/F182010.v4.tar'
# wget.download(night_image_url)


# # 2. Download Rwandan DHS and construct cluster-level aggregates
# 
# - **INPUT**: 
#   - `rwanda_clusters_location.csv`: Coordinates of the centroid of each cluster
# - **OUTPUT**: 
#   - `rwanda_cluster_avg_asset_2010.csv`: Comma-delimited file indicated average wealth of each cluster 
# 
# [Demographic and Health Surveys (DHS)](http://dhsprogram.com/What-We-Do/Survey-Types/DHS.cfm) are nationally-representative household surveys that provide data for a wide range of monitoring and impact evaluation indicators in the areas of population, health, and nutrition. For this, we will need to download the [2010 Rwandan DHS data](http://dhsprogram.com/what-we-do/survey/survey-display-364.cfm). Do not forget to request for the GPS dataset. 
# 

# result is shown below.
# <img src="figure/map1.png" alt="Map" style="width: 600px;"/>

# In[1]:


import pandas as pd
import numpy as np

file_name = 'data/DHS/RWHR61FL.DAT'
cluster_file = 'data/DHS/rwanda_clusters_location.csv'
cluster_all = []
wealth_all = []
with open(file_name) as f:
    for line in f:
        cluster = int(line[15:23])
        wealth = int(line[230:238]) / 100000.0
        cluster_all.append(cluster)
        wealth_all.append(wealth)

df = pd.DataFrame({'cluster': cluster_all, 'wlthindf': wealth_all})
cluster_avg_asset = df.groupby('cluster')['wlthindf'].median().reset_index()
df_location = pd.read_csv(cluster_file)[['DHSCLUST', 'LATNUM', 'LONGNUM']]
result = cluster_avg_asset.merge(df_location, how='inner', left_on='cluster', right_on='DHSCLUST')[['cluster', 'wlthindf', 'LATNUM', 'LONGNUM']]
result.rename(columns={'LATNUM': 'latitude', 'LONGNUM':'longitude'}, inplace=True)
result.to_csv('intermediate_files/rwanda_cluster_avg_asset_2010.csv', index=False)


# # 3. Test whether night lights data can predict wealth, as observed in DHS
# 
# Now that we have "ground truth" measures of average cluster wealth, our goal is to understand whether the nightlights data can be used to predict wealth. First, merge the DHS and nightlights data, and then fit a model of wealth on nightlights.
# 
# ## 3.1 Merge nightlights and DHS data at cluster level
# - **INPUT**: 
#  - `F182010.v4d_web.stable_lights.avg_vis.tif`: Nightlights data, from Step 1
#  - `rwanda_cluster_avg_asset_2010.csv`: DHS cluster averages, from Step 2
# - **OUTPUT**: Merged datasets
#  - `DHS_nightlights.csv`: Merged dataset with 492 rows, and 6 columns (one indicates average cluster wealth, 5 nightlights features)
#  - Scatterplot of nightlights vs. DHS wealth
# 
# Perform a "spatial join" to compute the average nighttime luminosity for each of the DHS clusters. To do this, we should take the average of the luminosity values for the nightlights locations surrounding the cluster centroid.
# 
# Save output as `DHS_nightlights.csv` and check that it is the same as the file we have provided.
# 

# In[1]:


import time
import os
import os.path
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy import misc
from io import StringIO
gdal.UseExceptions()
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import urllib
import pandas as pd
import numpy as np


def read_raster(raster_file):
    """
    Function
    --------
    read_raster

    Given a raster file, get the pixel size, pixel location, and pixel value

    Parameters
    ----------
    raster_file : string
        Path to the raster file

    Returns
    -------
    x_size : float
        Pixel size
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel
    centroid_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the centroid in each pixel
    centroid_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the centroid in each pixel
    bands_data : numpy.ndarray  shape: (number of rows, number of columns, 1)
        Pixel value
    """
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
    """
    Function
    --------
    get_cell_idx

    Given a point location and all the pixel locations of the raster file,
    get the column and row index of the point in the raster

    Parameters
    ----------
    lon : float
        Longitude of the point
    lat : float
        Latitude of the point
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel
    
    Returns
    -------
    lon_idx : int
        Column index
    lat_idx : int
        Row index
    """
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx


# In[2]:



raster_file = 'data/nighttime_image/F182010.v4d_web.stable_lights.avg_vis.tif'
x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data = read_raster(raster_file)

# save the result in compressed format - see https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
np.savez('intermediate_files/nightlight.npz', top_left_x_coords=top_left_x_coords, top_left_y_coords=top_left_y_coords, bands_data=bands_data)


# In[3]:



# get nightlight features for each cluster
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


clusters = pd.read_csv('intermediate_files/rwanda_cluster_avg_asset_2010.csv')
data_all = clusters.apply(lambda x: get_nightlight_feature([x['cluster'], x['wlthindf'], x['longitude'], x['latitude']]), axis=1)
data_all.to_csv('intermediate_files/DHS_nightlights.csv', index=None)


# In[4]:


import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

ax = sns.regplot(x="mean_", y="wealth", data=data_all)
plt.xlabel('Average nighttime luminosity')
plt.ylabel('Average cluster wealth')
plt.xlim([0, 50])


# ## 3.2. Fit a model of wealth as a function of nightlights
# - **INPUT**: 
#  - `DHS_nightlights.csv`, from Step 3.1
# - **OUTPUT**: 
#  - R^2 of model
#  
# Above, fit a regression line to illustrate the relationship between cluster average wealth and corresponding cluster nightlights. Now, use [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29) to get a better sense of out of sample accuracy.
# 

# In[5]:


from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

data_all = pd.read_csv('intermediate_files/DHS_nightlights.csv')
data_all = data_all[['max_', 'min_', 'mean_', 'median_', 'std_', 'wealth']].values
np.random.seed(123)

alphas_list = np.logspace(-1, 5, 7)
final = []
for alpha in alphas_list:
    kf = KFold(n_splits=10, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(data_all):
        reg = Ridge(alpha=alpha)
        train = data_all[train_index]
        test = data_all[test_index]
        reg.fit(train[:, :-1], train[:, -1])
        s = reg.score(test[:, :-1], test[:, -1])
        scores.append(s)
    final.append(np.mean(scores))

print('R^2 of the best model: {:.3f}'.format(np.max(final)))


# # 4. Download daytime satellite imagery 
# - **INPUT**: 
#  - Google Maps API key
#  - `Sector_Boundary_2012.shp`: Rwandan shapefile
# - **OUTPUT**: 
#  - Thousands of satellite images (store in directory `google_image/`)
# 
# We will use the Google Static Maps API to download satellite images. Refer [Google Static Maps introduction](https://developers.google.com/maps/documentation/static-maps/intro) and [Google Static Maps API Usage Limits](https://developers.google.com/maps/documentation/static-maps/usage-limits). Must apply for an API key before downloading. 
# 
# Download the images from Google at zoom level 16 (pixel resolution is about 2.5m). Set the image size to be 400 pixels X 400 pixels, so that each image downloaded will cover 1 square kilometer. In this way, each daytime image downloaded will correspond to a single pixel from the nighttime imagery from Step 1 above.
# 

# ![title](figure/data_description.png)

# In[6]:


# Helper function to read a shapefile
def get_shp_extent(shp_file):
    """
    Function
    --------
    get_shp_extent

    Given a shapefile, get the extent (boundaries)

    Parameters
    ----------
    shp_file : string
        Path to the shapefile
    
    Returns
    -------
    extent : tuple
        Boundary location of the shapefile (x_min, x_max, y_min, y_max)
    """
    print("shp_file_ext")
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(inShapefile, 0)
    inLayer = inDataSource.GetLayer()
    extent = inLayer.GetExtent()
    # x_min_shp, x_max_shp, y_min_shp, y_max_shp = extent
    return extent


# In[ ]:


# Helper functions to download images from Google Maps API

from retrying import retry
import urllib.request

@retry(wait_exponential_multiplier=1000, wait_exponential_max=3600000)
def save_img(url, file_path, file_name):
    """
    Function
    --------
    save_img

    Given a url of the map, save the image

    Parameters
    ----------
    url : string
        URL of the map from Google Map Static API
    file_path : string
        Folder name of the map
    file_name : string
        File name
    
    Returns
    -------
    None
    """
    
    a = urllib.request.urlopen(url).read()
    
    b = StringIO.StringIO(a)
    image = ndimage.imread(b, mode='RGB')
    # when no image exists, api will return an image with the same color. 
    # and in the center of the image, it said'Sorry. We have no imagery here'.
    # we should drop these images if large area of the image has the same color.
    if np.array_equal(image[:,:10,:],image[:,10:20,:]):
        print("=======")
        pass
    else:
        print('---')
        misc.imsave(file_path + file_name, image[50:450, :, :])

# Now read in the shapefile for Rwanda and extract the edges of the country
inShapefile = "data/shp/Sector_Boundary_2012/Sector_Boundary_2012.shp"
x_min_shp, x_max_shp, y_min_shp, y_max_shp = get_shp_extent(inShapefile)

left_idx, top_idx = get_cell_idx(x_min_shp, y_max_shp, top_left_x_coords, top_left_y_coords)
right_idx, bottom_idx = get_cell_idx(x_max_shp, y_min_shp, top_left_x_coords, top_left_y_coords)

key = 'AIzaSyCC2HwHLNIfDZsUyz63R1bIGIl_HA3x50M'
m = 1
for i in range(left_idx, right_idx + 1):
    for j in range(top_idx, bottom_idx + 1):
        lon = centroid_x_coords[i]
        lat = centroid_y_coords[j]
        print(lon,lat)
        url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' +                str(lon) + '&zoom=16&size=400x500&maptype=satellite&key=' + key
        lightness = bands_data[j, i, 0]
        file_path = 'google_image/' + str(lightness) + '/'
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file_name = str(i) + '_' + str(j) +'.jpg'
        save_img(url, file_path, file_name)
        print(file_path, file_name)
        if m % 100 == 0:
            print(m)
        m += 1


# # 5. Test whether basic features of daytime imagery can predict wealth
# In step 3, we tested whether nightlight imagery could predict the wealth of Rwandan villages. We will now test whether daytime imagery can predict village wealth. Start by extracting simple metrics from the daytime imagery; in step 6 we will use more sophsticated methods to engineer these features from the images. 
# 
# ## 5.1. Extract "basic" features from daytime imagery
# - **INPUT**: 
#  - `google_image/...`: Raw images, from Step 4
# - **OUTPUT**: 
#  - `google_image_features_basic.csv`: Image features 
# 
# Convert the raw data from the satellite imagery into a set of features that can be used in a machine learning algorithm. 

# In[ ]:


images_name = []
for i in range(64):
    dir_ = 'data/google_image/' + str(i) + '/'
    image_files = os.listdir(dir_)
    images_name.append(image_files)


def get_image_basic_feature(image_file):
    image = ndimage.imread(image_file, mode='RGB')
    features = []
    for i in range(3):
        image_one_band = image[:, :, i].flatten()
        features.append(image_one_band)
    features = np.asarray(features)
    max_ = np.max(features, axis=1)
    min_ = np.min(features, axis=1)
    mean_ = np.mean(features, axis=1)
    median_ = np.median(features, axis=1)
    std_ = np.std(features, axis=1)
    return np.concatenate([max_, min_, mean_, median_, std_]).tolist()


feature_all = []
a = 0
t1 = time.time()
for i, images in enumerate(images_name):
    path = 'data/google_image/' + str(i) + '/'
    for image in images:
        x, y = [int(idx) for idx in image[:-4].split('_')]
        file_ = path + image
        feature = get_image_basic_feature(file_)
        feature = [x, y] + feature
        feature_all.append(feature)
        if a % 10000 == 0:
            t2 = time.time()
            print a
            print t2 - t1
            t1 = time.time()
        a += 1

feature_all = np.asarray(feature_all)
np.savetxt('intermediate_files/google_image_features_basic.csv', feature_all)


# ## 5.2. Merge daytime images with DHS data
# 
# - **INPUT**: 
#  - `google_image_features_basic.csv`: Satellite imagery features, from Step 5.1
#  - `rwanda_cluster_avg_asset_2010.csv`: DHS cluster averages, from Step 2
# - **OUTPUT**: Merged datasets
#  - `data/model/DHS_daytime.csv`: Merged dataset with 492 rows, and 16 columns (one indicates average cluster wealth, 15 daytime image features)
# 
# 

# In[ ]:



features_basic = np.loadtxt('intermediate_files/google_image_features_basic.csv')

def get_daytime_feature(sample):
    idx, wealth, x, y = sample
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    features_100 = []
    for i in xrange(left_idx, right_idx + 1):
        for j in xrange(up_idx, low_idx + 1):
            feature = features_basic[((features_basic[:,0]==i) & (features_basic[:,1]==j)),][:,2:]
            if len(feature) > 0:
                features_100.append(feature)
    if len(features_100) == 0:
        return np.asarray([np.nan] * 15 + [wealth]).tolist()
    else:
        features_all = np.concatenate(features_100, axis=0)
        mean_ = np.mean(features_all, axis=0).tolist()
        mean_.append(wealth)
        return mean_


clusters = pd.read_csv('intermediate_files/rwanda_cluster_avg_asset_2010.csv')
clusters['feature'] = clusters.apply(lambda x: get_daytime_feature([x['cluster'], x['wlthindf'], x['longitude'], x['latitude']]), axis=1)

data_all = clusters['feature']
data_all = np.asarray([i for i in data_all])
data_all = data_all[~np.isnan(data_all).any(axis=1)]

np.savetxt('intermediate_files/DHS_daytime.csv', data_all)


# ## 5.3. Fit a model of wealth as a function of basic daytime features
# - **INPUT**: 
#  - `data/model/DHS_daytime.csv`, from Step 5.2
# - **OUTPUT**: 
#  - R^2 of model
#  

# In[17]:



data_all = np.loadtxt('intermediate_files/DHS_daytime.csv')
alphas_list = np.logspace(-1, 5, 7)
final = []
for alpha in alphas_list:
    kf = KFold(n_splits=10, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(data_all):
        reg = Ridge(alpha=alpha)
        train = data_all[train_index]
        test = data_all[test_index]
        reg.fit(train[:, :-1], train[:, -1])
        s = reg.score(test[:, :-1], test[:, -1])
        scores.append(s)
    final.append(np.mean(scores))

print 'R^2 of the best model: {:.3f}'.format(np.max(final))


# # 6. Extract features from daytime imagery using deep learning libraries
# 
# 
# ## 6.1. Use the keras library to use a basic CNN to extract features of the daytime images 
#  
# - **INPUT**: 
#  - `google_image/...`: Raw images, from Step 4
# - **OUTPUT**: 
#  - `google_image_features_cnn.csv`: Image features 
# 
# Begin by using a Convolutional Neural Network that has been pre-trained on ImageNet to extract features from the images. We recommend using the [`Keras` library](https://keras.io/), which provides a very straightforward interface to [TensorFlow](https://www.tensorflow.org/).
# 

# In[ ]:



from vgg16 import VGG16
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from multiprocessing import Pool
import os
import time
import pandas as pd
import numpy as np
from keras.models import Model


images_name = {} 
for i in range(64):
    dir_ = 'data/google_image/' + str(i) + '/'
    image_files = os.listdir(dir_)
    for f in image_files:
        images_name[f] = i


def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx


npzfile = np.load('intermediate_files/nightlight.npz')
print npzfile.files
top_left_x_coords = npzfile['top_left_x_coords']
top_left_y_coords = npzfile['top_left_y_coords']
bands_data = npzfile['bands_data']

# get image featuers
base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)


def get_input_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]


def get_daytime_feature(sample):
    idx, wealth, x, y = sample
    print idx
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    features_100 = []
    for i in xrange(left_idx, right_idx + 1):
        for j in xrange(up_idx, low_idx + 1):
            file_name = str(i) + '_' + str(j) + '.jpg'
            if file_name in images_name:
                luminosity = images_name[file_name]
                feature = get_input_feature('data/google_image/' + str(luminosity) + '/' + file_name)
                features_100.append(feature)
    if len(features_100) == 0:
        print 'nononono: ' + str(idx)
        return np.asarray([np.nan] * 4096 + [wealth]).tolist()
    else:
        features_100 = np.asarray(features_100)
        return np.append(np.mean(features_100, axis=0), wealth).tolist()


clusters = pd.read_csv('intermediate_files/rwanda_cluster_avg_asset_2010.csv')
clusters['feature'] = clusters.apply(lambda x: get_daytime_feature([x['cluster'], x['wlthindf'], x['longitude'], x['latitude']]), axis=1)

data_all = clusters['feature']
data_all = np.asarray([i for i in data_all])
data_all = data_all[~np.isnan(data_all).any(axis=1)]

np.savetxt('intermediate_files/google_image_features_cnn.csv', data_all)


# # 6.2. Test whether these new features of satellite imagery can predict wealth
# - **INPUT**: 
#  - `google_image_features_cnn.csv`: Satellite imagery features, from Step 6.1
#  - `rwanda_cluster_avg_asset_2010.csv`: DHS cluster averages, from Step 2
# - **OUTPUT**: 
#  - `data/model/DHS_daytime.csv`: Merged dataset with 492 rows, and 4097 columns (one indicates average cluster wealth, 4096 CNN-based features)
#  - R^2 of model
#  

# In[5]:


from sklearn.decomposition import PCA

data_all = np.loadtxt('intermediate_files/google_image_features_cnn.csv')
x = data_all[:, :-1]
y = data_all[:, -1]
pca = PCA(n_components=1)
pca.fit(x)
proj = pca.transform(x)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(proj[:,0], y, 'o', c='blue', markersize=5, markeredgecolor='none', alpha=0.5)
plt.xlabel('PCA 1')
plt.ylabel('Average wealth index')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


# In[13]:


alphas_list = np.logspace(-1, 5, 7)
final = []
for alpha in alphas_list:
    kf = KFold(n_splits=10, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(data_all):
        reg = Ridge(alpha=alpha)
        train = data_all[train_index]
        test = data_all[test_index]
        train_x = train[:, :-1]
        train_y = train[:, -1]
        test_x = test[:, :-1]
        test_y = test[:, -1]
        # reduce dimensions to avoid overfitting
        pca = PCA(n_components=100)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)
        reg.fit(train_x, train_y)
        s = reg.score(test_x, test_y)
        scores.append(s)
    final.append(np.mean(scores))

print 'R^2 of the best model: {:.3f}'.format(np.max(final))


# # 7. Extra Things
# 
# The only thing missing at this point is the "transfer learning" step. In other words, instead of using the image features extracted by the CNN directly, we want to retrain the CNN to predict nightlights from daytime imagery, and use those features, which presumably are more appropriate to our final prediction task.
# 
# ## 7.1. Use the nightlights to retrain the CNN and extract features
# 
# - **INPUT**: 
#  - `google_image/...`: Raw images, from Step 4
# - **OUTPUT**: 
#  - `google_image_features_cnn_retrained.csv`: Image features 
# 
# Following the approach used in the paper, first divide daytime images into three groups, corresponding to images where the corresponding night-lights pixel is dim, medium, or bright. Use these values to define groups: [0, 3), [3, 35), [35, 64). 

# In[ ]:



def move_to_group(lightness_small, lightness_big, class_id):
    new_directory = 'data/google_image_cnn/class_' + str(class_id) + '/'
    if not os.path.isdir(new_directory):
        os.makedirs(new_directory)
    for i in range(lightness_small, lightness_big):
        path = 'data/google_image/' + str(i) + '/'
        for f in os.listdir(path):
            copyfile(path + f, new_directory + f)

move_to_group(0, 3, 1)
move_to_group(3, 35, 2)
move_to_group(35, 64, 3)


# ## 7.2. Test whether "deep" features of satellite imagery can predict wealth
# - **INPUT**: 
#  - `google_image_cnn/...`: Satellite images from 7.1
# - **OUTPUT**: 
#  - `data/model/DHS_CNN.csv`: Merged dataset with 492 rows, and 4097 columns (one indicates average cluster wealth, 4096 CNN features)
#  - R^2 of model
# 

# In[ ]:



from vgg16 import VGG16
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.models import Model
from multiprocessing import Pool
import os
import time


# get image featuers

model_old = VGG16(weights='imagenet', include_top=False)


def get_input_feature(img_path):
    # img = image.load_img(img_path, target_size=(400, 400))
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_old.predict(x)
    return features[0]


# train
all_figures = []
trainLabels = []

# need upsampling because of the unbalance of the training classes
path_1 = 'google_image_cnn_upsampling/train/class_1/'
# path_1 = 'google_image_cnn/class_1/'
class_1_files = os.listdir(path_1)
trainLabels += [[1, 0, 0]] * len(class_1_files)
all_figures += [path_1 + i for i in class_1_files]

path_2 = 'google_image_cnn_upsampling/train/class_2/'
# path_2 = 'google_image_cnn/class_2/'
class_2_files = os.listdir(path_2)
trainLabels += [[0, 1, 0]] * len(class_2_files)
all_figures += [path_2 + i for i in class_2_files]

path_3 = 'google_image_cnn_upsampling/train/class_3/'
# path_3 = 'google_image_cnn/class_3/'
class_3_files = os.listdir(path_3)
trainLabels += [[0, 0, 1]] * len(class_3_files)
all_figures += [path_3 + i for i in class_3_files]


# a = get_input_feature(all_figures[0])
# pool = Pool(10)
# trainData = pool.map(get_input_feature, all_figures)

trainData = []
t1 = time.time()
for idx, i in enumerate(all_figures):
    a = get_input_feature(i)
    trainData.append(a)
    if idx % 1000 == 0:
        t2 = time.time()
        print idx
        print t2 - t1
        t1 = time.time()


x_all = np.asarray(trainData)
y_all = np.asarray(trainLabels)


# test
all_figures = []
testLabels = []

path_1 = 'google_image_cnn_upsampling/test/class_1/'
# path_1 = 'google_image_cnn/class_1/'
class_1_files = os.listdir(path_1)
testLabels += [[1, 0, 0]] * len(class_1_files)
all_figures += [path_1 + i for i in class_1_files]

path_2 = 'google_image_cnn_upsampling/test/class_2/'
# path_2 = 'google_image_cnn/class_2/'
class_2_files = os.listdir(path_2)
testLabels += [[0, 1, 0]] * len(class_2_files)
all_figures += [path_2 + i for i in class_2_files]

path_3 = 'google_image_cnn_upsampling/test/class_3/'
# path_3 = 'google_image_cnn/class_3/'
class_3_files = os.listdir(path_3)
testLabels += [[0, 0, 1]] * len(class_3_files)
all_figures += [path_3 + i for i in class_3_files]


# a = get_input_feature(all_figures[0])
# pool = Pool(10)
# testData = pool.map(get_input_feature, all_figures)

testData = []
t1 = time.time()
for idx, i in enumerate(all_figures):
    a = get_input_feature(i)
    testData.append(a)
    if idx % 1000 == 0:
        t2 = time.time()
        print idx
        print t2 - t1
        t1 = time.time()


x_all_test = np.asarray(testData)
y_all_test = np.asarray(testLabels)



# np.savez('google_image_feature.npz', x_all=x_all, y_all=y_all)
np.savez('google_image_feature_upsampling.npz', x_all=x_all, y_all=y_all, 
         x_all_test=x_all_test, y_all_test=y_all_test)

# npzfile = np.load('google_image_feature_upsampling.npz')
# print npzfile.files
# x_all = npzfile['x_all']
# y_all = npzfile['y_all']


x_train = x_all
x_test = x_all_test
y_train = y_all
y_test = y_all_test


# the model configuration: 
# https://github.com/nealjean/predicting-poverty/blob/master/model/predicting_poverty_deploy.prototxt
model = Sequential()
model.add(Convolution2D(4096, 6, 6, activation='relu', input_shape=(12, 12, 512), subsample=(6, 6), name='input'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, 1, 1, activation='relu', subsample=(1, 1), name='conv_7'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, 1, 1, subsample=(1, 1), name='conv_8'))
model.add(AveragePooling2D((2, 2), strides=(1, 1), name='add_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(3))
model.add(Activation("softmax"))

opt = SGD(lr=1e-2)
# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


model.fit(x_train, y_train, batch_size=100, nb_epoch=10, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)  # 0.778 after 4 epoch
print score


# In[ ]:


# get features
npzfile = np.load('intermediate_files/nightlight.npz')
print npzfile.files
top_left_x_coords = npzfile['top_left_x_coords']
top_left_y_coords = npzfile['top_left_y_coords']
bands_data = npzfile['bands_data']


def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx


model_select = Model(input=model.input, output=model.get_layer('add_pool').output)


images_name = {} 
for i in range(64):
    dir_ = 'data/google_image/' + str(i) + '/'
    image_files = os.listdir(dir_)
    for f in image_files:
        images_name[f] = i


def get_input_feature_2(img_path):
    # img = image.load_img(img_path, target_size=(400, 400))
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_old.predict(x)
    pool_features = model_select.predict(features)
    return pool_features[0]


def get_daytime_feature(sample):
    idx, wealth, x, y = sample
    print idx
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    features_100 = []
    for i in xrange(left_idx, right_idx + 1):
        for j in xrange(up_idx, low_idx + 1):
            file_name = str(i) + '_' + str(j) + '.jpg'
            if file_name in images_name:
                luminosity = images_name[file_name]
                feature = get_input_feature_2('data/google_image/' + str(luminosity) + '/' + file_name)
                features_100.append(feature)
    if len(features_100) == 0:
        print 'nononono: ' + str(idx)
        return np.asarray([np.nan] * 4096 + [wealth]).tolist()
    else:
        features_100 = np.asarray(features_100)
        return np.append(np.mean(features_100, axis=0), wealth).tolist()


clusters = pd.read_csv('intermediate_files/rwanda_cluster_avg_asset_2010.csv')
clusters['feature'] = clusters.apply(lambda x: get_daytime_feature([x['cluster'], x['wlthindf'], 
                                                                    x['longitude'], x['latitude']]), axis=1)

data_all = clusters['feature']
data_all = np.asarray([i for i in data_all])
data_all = data_all[~np.isnan(data_all).any(axis=1)]

np.savetxt('intermediate_files/google_image_features_cnn_retrain.csv', data_all)


# In[17]:


data_all = np.loadtxt('intermediate_files/google_image_features_cnn_retrain.csv')

alphas_list = np.logspace(-1, 5, 7)

final = []
for alpha in alphas_list:
    kf = KFold(n_splits=10, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(data_all):
        reg = Ridge(alpha=alpha)
        train = data_all[train_index]
        test = data_all[test_index]
        train_x = train[:, :-1]
        train_y = train[:, -1]
        test_x = test[:, :-1]
        test_y = test[:, -1]
        # reduce dimensions to avoid overfitting
        pca = PCA(n_components=100)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)
        reg.fit(train_x, train_y)
        s = reg.score(test_x, test_y)
        scores.append(s)
    final.append(np.mean(scores))

print 'R^2 of the best model: {:.3f}'.format(np.max(final))


# # 8. Construct a high-resolution map of the distribution of predicted wealth
# - **INPUT**: 
#  - Model, image features (data/model/features_all_predictimage_location.csv)
# - **OUTPUT**: 
#  - Map ('poverty_mapping.tiff')
#  
# Create a heatmap showing the distribution of predicted wealth in Rwanda. With any luck, it will look something like this:
# <img src="figure/pmap.png" alt="Map"/>

# In[ ]:



def array_to_raster(x_size, y_size, left, top, bands_data, no_data_value, filename):
    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        filename,
        bands_data.shape[1],
        bands_data.shape[0],
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform((
        left,
        x_size,
        0,
        top,
        0,
        y_size))

    # dataset.SetProjection(wgs84)
    outband = dataset.GetRasterBand(1)
    outband.WriteArray(bands_data[:, :, 0])
    outband.FlushCache()  # Write to disk.
    outband.SetNoDataValue(no_data_value)


features_all_predict = np.genfromtxt('data/model/features_all_predict.csv', delimiter=',')
image_loc = np.genfromtxt('data/model/image_loc.csv', delimiter=',')

x_image_num = right_idx + 1 - left_idx
y_image_num = bottom_idx + 1 - top_idx

output_array = np.ndarray(shape=(y_image_num, x_image_num, 1))
no_data_value = 100
output_array.fill(no_data_value)

for idx, v in enumerate(image_loc):
    x, y = v
    x_id = x - left_idx
    y_id = y - top_idx
    output_array[y_id, x_id, 0] = features_all_predict[idx]

array_to_raster(x_size, - x_size, top_left_x_coords[left_idx],
                top_left_y_coords[top_idx], output_array, no_data_value, 'poverty_mapping.tiff')

