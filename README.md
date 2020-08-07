# Gender Detection from Facial Images with and without Mask
### Update

August 7th: visualization function for Loss and Accuracy added.

### Author

* Soo Woon (Shawn) Chung
* Jingwen (Rebecca) Du
* Jingxian (Phebe) Lan
* Zhongyuan (Jasper) Zhang

### Introduction

Here is a collection of code and datasets corresponding to our project "Gender Detection from Facial Images with and without Mask".



Due to size limit, The Google Shared Folder https://drive.google.com/drive/folders/14a38EqwCzzJNp3faOuxFNfnFgyEW-hPO?usp=sharing includes the following files:

**Datasets**: CelebA, Casia WebFace Facial, Google Crawled

**Best Model's Weight** :weights6.best.inc.male.hdf5

Otherwise, all required files are stored in the github repository.

Before training the model, Unzip datasets in to `imgs/` folder.

The directory structure is as follows:
```
genderdetect/
	|-> main.py/					* changing hyperparameters and set folder path
	|-> load_attr.py			* Load image info
	|-> traininig.py			* network strucutres, train/test functions
	|-> generate.py				* data preprocessing and loading
	|-> plot.py						* results visualization
```

