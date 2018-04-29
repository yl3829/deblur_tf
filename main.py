import tensorflow as tf
import numpy as np
from model_tf import deblur_model
import argparse
from utlis import load_images

if __name__ == '__main__':
    
    # add argument
    parser = argparse.ArgumentParser(description="deblur train")
    parser.add_argument(
        "--is_train", help="train or generate",
        default=0,type=int)
    

    param = parser.parse_args()
    
    # load data
    train_data = load_images('./images/train',n_images=-1)
    test_data = load_images('./images/test', n_images=200)
    
    
    print('Building model')
    model = deblur_model(param)
    
    if param.is_train:
        print('Training model')
        model.train(data)
    else:
        print('Debluring')
        model.generate(data)
    