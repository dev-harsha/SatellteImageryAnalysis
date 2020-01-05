import os
from shutil import copyfile

def move_to_group(lightness_small, lightness_big, class_id):
    new_directory = 'data/google_image_cnn/class_' + str(class_id) + '/'
    if not os.path.isdir(new_directory):
        os.makedirs(new_directory)
    for i in range(lightness_small, lightness_big):
        path = 'data/google_image/' + str(i) + '/'
        for f in os.listdir(path):
            print(path+f, new_directory+f)
            copyfile(path + f, new_directory + f)

move_to_group(0, 3, 1)
move_to_group(3, 35, 2)
move_to_group(35, 64, 3)