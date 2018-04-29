import tensorflow as tf
import numpy as np
from model_tf import deblur_model
import argparse
from utils import load_images
import os

if __name__ == '__main__':
    
    # add argument
    parser = argparse.ArgumentParser(description="deblur train")
    parser.add_argument(
        "--is_train", help="train or generate",
        default=1,type=int)
    parser.add_argument(
        "--image_dir", help="Path to the image",
        default="./images")
    parser.add_argument(
        "--g_input_size", help="Generator input size of the image",
        default=256,type=int)
    #parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsampling in generator')
    parser.add_argument('--n_blocks_gen', type=int, default=9, help='# of res block in generator')
    parser.add_argument('--d_input_size', type=int, default=256, help='Generator input size')
    parser.add_argument('--kernel_size', type=int, default=4, help='kernel size factor in discriminator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    '''
    input_size = self.param.g_input_size
    ngf = self.param.ngf
    n_downsampling =  self.param.n_downsampling
    output_nc = self.param.output_nc
    n_blocks_gen = self.param.n_blocks_gen
    
    input_size = self.param.d_input_size
    ndf = self.param.ndf
    kernel_size = self.param.kernel_size
    n_layers = self.param.n_layers
    '''
    
    param = parser.parse_args()
    
    # load data
    train_data = load_images(os.path.join(param.image_dir, "train"),n_images=-1)
    test_data = load_images(os.path.join(param.image_dir, "train"), n_images=200)
    
    
    print('Building model')
    model = deblur_model(param)
    
    if param.is_train:
        print('Training model')
        model.train(train_data)
    else:
        print('Debluring')
        model.generate(test_data, trained_model)
    