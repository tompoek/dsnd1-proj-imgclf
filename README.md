# Project Description

Build flower image classifier with PyTorch, then convert it into command line application... Here's the [original repo](https://github.com/udacity/aipnd-project).

## Setup

* Python
* Numpy
* Pytorch (with Torchvision)
* Pillow
* Jupyter Notebook
* Matplotlib and Seaborn (optional)

## How to Use

### Notebook Lab

Open Image_Classifier_Project.ipynb

### Command Line

Train model: (where "data_dir" is your image folder)
```sh
$ python train.py data_dir
```

Classify image: (where "input" and "checkpoint" are the path to a specific image and trained model checkpoint respectively)
```sh
$ python predict.py input checkpoint
```

### Image Data

Images must be stored in this pattern for training, validation and testing, where the integer subfolders are properly mapped with categories.
```bash
image_folder
├── train
│   ├── 0
│   ├── 1
│   │   ├── xxx.jpg
│   │   ├── ......
│   │   └── xxx.jpg
│   └── ......
├── valid
│   ├── 0
│   ├── 1
│   └── ......
└── test
    ├── 0
    └── ......
```

### Notes to Reviewer

Could you help figure out how to wrap MyClassifier() constructor to be flexible on number of hidden layers (not only units)? Thanks!!
