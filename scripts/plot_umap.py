from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
import cv2
import umap
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


class UMAP:
    def __init__(self):
        pass

    
    def _load_images_from_folder_2d(self, folder_path):
        images = []
        labels = []
        for class_name in os.listdir(folder_path):
            class_folder = os.path.join(folder_path, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, filename)
                    try:
                        img = Image.open(img_path).convert('RGB').resize((64, 64))
                        images.append(np.array(img).flatten())
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        return images, labels


    def _umap_engine_2d(self, folder_path, title, legend, save_fig, save_dir, save_name, dpi, width, height, cmap):
        
        images, labels = self._load_images_from_folder_2d(folder_path)
        
        if images and labels:    
            images = StandardScaler().fit_transform(images)
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(images)

            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
            unique_labels = sorted(set(labels))
            num_classes = len(unique_labels)

            fig, ax = plt.subplots(figsize=(width, height))
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=numeric_labels, cmap=cmap, s=4)
            ax.set_aspect('equal', 'datalim')
            ax.set_title(f'{title}', fontsize=13, weight='bold')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if legend == True:
                cbar = fig.colorbar(sc, boundaries=np.arange(num_classes + 1) - 0.5, ax=ax)
                cbar.set_ticks(np.arange(num_classes))

                # Adjust font size and rotation for legend text
                cbar.ax.tick_params(labelsize=12, pad=8)  # You can adjust the fontsize and padding values
                cbar.set_ticklabels(unique_labels, rotation=90)

                # Center-align the legend text
                cbar.ax.set_yticklabels(unique_labels, va='center')
                cbar.ax.yaxis.set_ticks_position('right') 

            plt.show()
            
            if save_fig==True:
                fig.savefig(f'{save_dir}/{save_name}.jpg', dpi=dpi, bbox_inches='tight')
        else:
            print("No images were found in the specified folder.")
        
        
    def plot_2d(self, folder_path, title='', legend=True, save_fig=False, save_dir='', save_name='', dpi=600, width=5, height=4,
             cmap='coolwarm'):
        
        self._umap_engine_2d(
            folder_path=folder_path,
            title=title,
            legend=legend,
            save_fig=save_fig,
            save_dir=save_dir,
            save_name=save_name,
            dpi=dpi,
            width=width,
            height=height,
            cmap=cmap
        )

    
    def _load_images_from_folder_3d(self, folder, start_limit, end_limit):
        data = []
        labels = []
        for subdir, dirs, files in os.walk(folder):
            # reset limit count
            start = 0
            end = 0
            count = 0

            # check limit
            if start_limit==0:
                for file in files:
                    count = count + 1
                    if count < 4000:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (64, 64)).flatten()
                            data.append(img)
                            labels.append(os.path.basename(subdir))
            else:
                for file in files:
                    start = start + 1
                    if start_limit < start < end_limit:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (64, 64)).flatten()
                            data.append(img)
                            labels.append(os.path.basename(subdir))

        return np.array(data), labels
    
    
    def _umap_engine_3d(self, folder_path, start_limit, end_limit, title, legend, save_fig, save_dir, save_name, dpi, angle_1, angle_2):
        
        data, labels = self._load_images_from_folder_3d(folder_path, start_limit, end_limit)
        reduced_data = umap.UMAP(n_components=3, random_state=42).fit_transform(data)
        
        sns.set(style='white')
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(angle_1, angle_2)

        unique_labels = np.unique(labels)
        for label in unique_labels:
            selected_data = reduced_data[np.where(np.array(labels) == label)]
            ax.scatter(selected_data[:, 0], selected_data[:, 1], selected_data[:, 2], label=label)

        if legend==True:
            legend = ax.legend(fontsize=25, ncol=4, bbox_to_anchor=(1, 1.1))
            # Make the border and background of the legend invisible
            legend.get_frame().set_linewidth(0)
            legend.legendPatch.set_alpha(0)
            # Increase the size of the dots
            for handle in legend.legend_handles:
                handle.set_sizes([500])

        ax.set_title(f'{title}', fontsize=35, loc='center', y=-0.1, weight='bold')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        plt.show()
        
        if save_fig==True:
                fig.savefig(f'{save_dir}/{save_name}.jpg', dpi=dpi, bbox_inches='tight')
        
        
    def plot_3d(self, folder_path, start_limit=0, end_limit=0, title='', legend=True, save_fig=False, save_dir='', save_name='', dpi=150,
                angle_1=30, angle_2=45):
        
        self._umap_engine_3d(
            folder_path=folder_path, 
            title=title,
            save_fig=save_fig, 
            save_dir=save_dir,
            save_name=save_name, 
            legend=legend,
            angle_1=angle_1, 
            angle_2=angle_2,
            dpi=dpi,
            start_limit=start_limit, 
            end_limit=end_limit
        )