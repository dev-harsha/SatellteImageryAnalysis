from datetime import datetime
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
import urllib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


images_name = []
for i in range(64):
    dir_ = 'data/google_image/' + str(i) + '/'
    if os.path.exists(dir_):
        image_files = os.listdir(dir_)        
        images_name.append(image_files)
    else:
        images_name.append([])

def get_image_basic_feature(image_file):
    #image = ndimage.imread(image_file, mode='RGB')
    image = plt.imread(image_file)
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

today_date = date.today()
date_format = today_date.strftime("%d%m%Y")


log_file=open("csv_log_" + date_format + ".txt","a")
log_file.write(str(datetime.now()) + " ---------- New Logs ------------\n")

feature_all = []
a = 0
t1 = time.time()
for i, images in enumerate(images_name):
    path = 'data/google_image/' + str(i) + '/'
    print(i)
    for image in images:
        x, y = [int(idx) for idx in image[:-4].split('_')]
        file_ = path + image
        
        if os.path.exists(file_):
            print(i, file_)
            print(file_)
            log_file.write(file_ + "\n")
            feature = get_image_basic_feature(file_)
            print("pocessed...\n")
            log_file.write("pocessed...\n")
            feature = [x, y] + feature
            feature_all.append(feature)
            if a % 10000 == 0:
                t2 = time.time()
                print(a)
                print(t2 - t1)
                log_file.write(str(a))
                log_file.write(str(t2 - t1))
                t1 = time.time()
            a += 1

feature_all = np.asarray(feature_all)
np.savetxt('intermediate_files/google_image_features_basic.csv', feature_all, delimiter=",")