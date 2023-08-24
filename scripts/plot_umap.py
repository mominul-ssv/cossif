from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
import umap
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


class UMAP2D:
    def __init__(self):
        pass

    
    def _load_images_from_folder(self, folder_path):
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


    def _umap_engine(self, folder_path, title, legend, save_fig, save_dir, width, height, cmap):
        
        images, labels = self._load_images_from_folder(folder_path)
        
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
                cbar.ax.tick_params(labelsize=10, pad=8)  # You can adjust the fontsize and padding values
                cbar.set_ticklabels(unique_labels, rotation=90)

                # Center-align the legend text
                cbar.ax.set_yticklabels(unique_labels, va='center')
                cbar.ax.yaxis.set_ticks_position('right') 

            plt.show()
            
            if save_fig==True:
                fig.savefig(f'{save_dir}.jpg', dpi=600, bbox_inches='tight')
        else:
            print("No images were found in the specified folder.")
        
        
    def plot(self, folder_path, title='', legend=True, save_fig=False, save_dir='', width=5, height=4, cmap='coolwarm'):
        
        self._umap_engine(
            folder_path=folder_path,
            title=title,
            legend=legend,
            save_fig=save_fig,
            save_dir=save_dir,
            width=width,
            height=height,
            cmap=cmap
        )        