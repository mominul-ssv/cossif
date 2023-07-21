import os
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def generate_images(source_path, save_path, images_needed, batch_size=32, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0):
    
    temp_dir = os.path.join(save_path, 'temp_dir')
    os.makedirs(temp_dir)
    
    temp_class = os.path.join(temp_dir, 'temp_class')
    os.makedirs(temp_class)
    
    for file_name in os.listdir(source_path):
        source = os.path.join(source_path, file_name)
        destination = os.path.join(temp_class, file_name)
        shutil.copyfile(source, destination)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest'
    )
    
    aug_datagen = datagen.flow_from_directory(
        directory=temp_dir,
        save_to_dir=save_path,
        save_format='jpg',
        target_size=(256, 256)
    )
    
    num_batches = int(np.ceil(images_needed / batch_size))
    
    for i in tqdm(range(num_batches), colour="blue"):
        next(aug_datagen)
        
    shutil.rmtree(temp_dir)
    
    for file_name in tqdm(os.listdir(source_path), colour="magenta"):
        source = os.path.join(source_path, file_name)
        destination = os.path.join(save_path, file_name)
        shutil.copyfile(source, destination)