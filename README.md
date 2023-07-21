# CosSIF: Cosine Similarity-based Image Filtering
This is the official code for our CosSIF paper:<br/>[CosSIF: Cosine Similarity-based Image Filtering to overcome low
inter-class variation in synthetic medical image datasets]()

<p align="left">
<img src="plots/outputs/Pipeline.png" alt="image_description" style="padding: 5px" width="65%">
</p>

## 1. Specification of dependencies
This code requires two separate conda environments. Run the following to install the required packages on Windows.

```python
cd environments/
conda env create -f csf-ovs.yml
conda env create -f csf-model.yml
```

The following notebooks require the `csf-ovs` environment to be activated.

```
scripts/
|-- HAM10000/
|---- HAM10000_GAN_train.ipynb
|---- HAM10000_augment_ds.ipynb
|---- HAM10000_oversampling.ipynb
|---- HAM10000_pre-processing.ipynb
|---- HAM10000_filtering/
|------ HAM10000_FAGT.ipynb
|------ HAM10000_FBGT.ipynb
|-- ISIC-2016/
|---- ISIC-2016_GAN_train.ipynb
|---- ISIC-2016_augment_ds.ipynb
|---- ISIC-2016_oversampling.ipynb
|---- ISIC-2016_pre-processing.ipynb
|---- ISIC-2016_filtering/
|------ ISIC-2016_FAGT.ipynb
|------ ISIC-2016_FBGT.ipynb
```

Run the following to activate the `csf-ovs` environment.

```python
conda activate csf-ovs
```

The rest of the notebooks require the `csf-model` environment to be activated. Run the following to activate the `csf-model` environment.

```python
conda activate csf-model
```

The conda environments can also be setup manually. Install the packages mentioned `packages.txt` file.

Install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/older-downloads/) along with the following workloads:
- Desktop development with C++
- Universal Windows Platform development

## 2a. Datasets
Download and extract the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and [ISIC-2016](https://challenge.isic-archive.com/data/) (Task-3) ```datasets```, along with their associated ground truth files. Then, create a folder named ```datasets``` in the root directory and organize the folders and ground truth files in the following order:

```
datasets/
|-- HAM10000/
|---- downloads/
|------ HAM10000_images/
|------ HAM10000_metadata.csv
|-- ISIC-2016/
|---- downloads/
|------ ISBI2016_ISIC_Part3_Training_Data/
|------ ISBI2016_ISIC_Part3_Test_Data/
|-------ISBI2016_ISIC_Part3_Training_GroundTruth.csv
|-------ISBI2016_ISIC_Part3_Test_GroundTruth.csv
```

Note that you may need to rename the file and folder names as required.

## 2b. Pre-processing
To begin the `pre-processing`, run the following scripts associated with each dataset:

```
ISIC-2016_pre-processing.ipynb
```
```
HAM10000_pre-processing.ipynb
```

## 2c. Oversampling & filtering
For `oversampling`, run the following scripts associated with each dataset:

```
ISIC-2016_oversampling.ipynb
```
```
HAM10000_oversampling.ipynb
```

During the execution of the `oversampling` scripts, there comes a certain point where it becomes necessary to also execute the `filtering` and `GAN` training scripts.

For `filtering`, run the following scripts associated with each dataset:

```
ISIC-2016_FBGT.ipynb
ISIC-2016_FAGT.ipynb
```
```
HAM10000_FBGT.ipynb
HAM10000_FAGT.ipynb
```

For `GAN` training, run the following scripts associated with each dataset on [Kaggle](https://www.kaggle.com/):

```
ISIC-2016_GAN_train.ipynb
```
```
HAM10000_GAN_train.ipynb
```

Note that the saving of the trained `GANs` on [Kaggle](https://www.kaggle.com/) must be conducted manually. The details are provided on the scripts and must be followed accordingly.

## 2d. Dataset Augmentation
To achieve the final `augmented` datasets associated with each different experiment, run the following codes.

```
ISIC-2016_augment_ds.ipynb
```
```
HAM10000_augment_ds.ipynb
```

## 2e. Training Models
To train the `models` on the oversampled ISIC-2016 dataset and its filtered variants, navigate to the following directory and run the scripts. 

```
scripts/
|-- ISIC-2016/
|---- ISIC-2016_model_train/
|------ ConvNeXt/
|------ Swin-Transformer/
|------ ViT/
```

To train the `models` on the oversampled HAM10000 dataset and its filtered variants, navigate to the following directory and upload the scripts on [Kaggle](https://www.kaggle.com/) along with its corresponding variants of the dataset. 

```
scripts/
|-- HAM10000/
|---- HAM10000_model_train/
|------ ConvNeXt/
|------ Swin-Transformer/
|------ ViT/
```

Note that the saving of the trained `models` on [Kaggle](https://www.kaggle.com/) must be conducted manually. The details are provided on the scripts and must be followed accordingly.

## 2f. Performance evaluation
To evaluate the `performance` of all trained models on the oversampled ISIC-2016 dataset and its variants, run the following script:

```
ISIC-2016_model_test.ipynb
```

To evaluate the `performance` of all trained models on the oversampled HAM10000 dataset and its variants, run the following script:

```
HAM10000_model_test.ipynb
```

## 3. Pre-trained models
We provide our `pre-trained` models on [GitHub Releases]() for reproducibility. Below are the best-performing `pre-trained` models.

| Dataset | Model | Filtering | Sensitivity (%) | False Negative | Download |
| -------- | -------- | -------- | :--------: | :--------: | -------- |
| ISIC-2016 | ViT | FAGT (α = 0.75) | **72.00** | **21** |[download]() |

| Dataset | Model | Filtering | Accuracy (%) | F1-score (%) | Download |
| -------- | -------- | -------- | :--------: | :--------: | -------- |
| HAM10000 | ConvNeXt | FAGT (α = 0.85) | **94.44** | **84.06** | [download]() |

## 4. Demo

## 5. Citation

### Acknowledgements
This code base is built on top of the following repositories:
- https://github.com/NVlabs/stylegan2-ada-pytorch

This code utilizes the following models for fine-tuning:
- https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
- https://huggingface.co/google/vit-base-patch16-224
- https://huggingface.co/facebook/convnext-tiny-224