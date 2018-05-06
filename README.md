# What is this repo ?

This repository is for the final project of ECBM6040: Neural Networks and Deep Learning Research, the group of DGAN. Implementation of deblur in Tensorflow 

Original paper [DeBlur GANs](https://arxiv.org/pdf/1711.07064.pdf). 


By:
Yilin Lyu	 *yl3832*
Yanjun Lin *yl3829*
Yao Chou  *yc3332*


## Dataset:

Get the [GOPRO dataset](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing), and extract it in the working directory. The directory name of dataset should be `GOPRO_Large`.

## Use the dataset:
```
python organize_dataset.py --dir_in=GOPRO_Large --dir_out=images
```

## Deblur customized images:

Save the images you want to deblur in a folder called `own` inside the `image` folder

```
python main.py --customized=1
```

## Train a GAN model:
```
python main.py 
```

## Test model and evaluation 
```
python main.py --is_train=0 --testing_image=-1 --model_name=./model/your_model_name
```

