import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_images_first_four(path):
    plt.figure(figsize=(12, 4), dpi=60)
    for i, file in enumerate(os.listdir(path)[0:4]):
        image_path = path + '/' + file
        img = mpimg.imread(image_path)
        plt.subplot(1, 4, i+1) 
        plt.imshow(img)        
        

def plot_images_last_four(path):
    plt.figure(figsize=(12, 4), dpi=60)
    for i, file in enumerate(os.listdir(path)[-4:]):
        image_path = path + '/' + file
        img = mpimg.imread(image_path)
        plt.subplot(1, 4, i+1) 
        plt.imshow(img)