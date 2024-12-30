# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Download data and extract training data
```
mkdir -p flowers
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
tar -xf flower_data.tar.gz -C flowers
```

## Training script
```shell
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH] [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS] [--gpu] data_dir

positional arguments:
  data_dir              path to the folder of images

options:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   path to save the model checkpoints. Default: checkpoints
  --arch ARCH           CNN Model Architecture (densenet or vgg). Default: densenet
  --learning_rate LEARNING_RATE
                        Learning rate. Default: 0.001
  --hidden_units HIDDEN_UNITS
                        Hidden units. Default: 512
  --epochs EPOCHS       Number of epochs. Default: 3
  --gpu                 Use GPU for training. Default: False
```

## Inference script
```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu] image_file checkpoint

positional arguments:
  image_file            path to the image file
  checkpoint            path to the model checkpoint

options:
  -h, --help            show this help message and exit
  --top_k TOP_K         top k classes
  --category_names CATEGORY_NAMES
                        category names
  --gpu                 Use GPU for inference

```