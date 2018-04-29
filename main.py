import tensorflow as tf
import numpy as np
from model_tf import deblur_model
import argparse

if __name__ == '__main__':
    
    # add argument
    parser = argparse.ArgumentParser(description="deblur train")
    parser.add_argument(
        "--is_train", help="train or generate",
        default=0,type=int)
    

    param = parser.parse_args()
    
    # load data
    # ...
    
    print('Building model')
    model = deblur_model(param)
    
    if param.is_train:
        print('Training model')
        model.train(data)
    else:
        print('Debluring')
        model.generate(data)
    