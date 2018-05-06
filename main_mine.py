import tensorflow as tf
import numpy as np
from model_tf_mine import deblur_model
import argparse
from utils import load_images,load_own_images
import os
import h5py

if __name__ == '__main__':
    
    # add argument
    parser = argparse.ArgumentParser(description="deblur train")
    parser.add_argument("--is_train", help="train or generate", default=1,type=int)
    parser.add_argument("--image_dir", help="Path to the image", default="./images")
    parser.add_argument("--g_input_size", help="Generator input size of the image", default=256,type=int)
    #parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsampling in generator')
    parser.add_argument('--n_blocks_gen', type=int, default=9, help='# of res block in generator')
    parser.add_argument('--d_input_size', type=int, default=256, help='Generator input size')
    parser.add_argument('--kernel_size', type=int, default=4, help='kernel size factor in discriminator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size when training')
    parser.add_argument('--model_name',default=None, help='The pre-trained model name')
    parser.add_argument('--training_image',default=-1, type=int, help='number of image in training')
    parser.add_argument('--testing_image',default=200, type=int, help='number of image in testing')
    parser.add_argument('--save_freq',default=100, type=int, help='Model save frequency')
    parser.add_argument('--epoch_num',default=10, type=int, help='Number of epoch for training')
    parser.add_argument('--generate_image_freq', default=50, type=int, help='Number of iteration to generate image for checking')
    parser.add_argument('--is_generating', default=0, type=int, help='Generating own images')
    
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
    if param.is_generating:
        print('Generating own images')
        print('Building model')
        model = deblur_model(param)
        own_data=load_own_images(os.path.join(param.image_dir, "own"), n_images=-1)
        model.generate_own(own_data, trained_model=param.model_name)
    else:
        # load data
        print('Loading data')
        cache_file = 'train_cache.hdf5'
        if param.is_train:
            if os.path.exists(cache_file):
                h5f = h5py.File(cache_file,'r')
                train_data = {'A':h5f['A'][:], 'B':h5f['B'][:]}
            else:
                train_data = load_images(os.path.join(param.image_dir, "train"),n_images=param.training_image)
                h5f = h5py.File(cache_file, 'w')
                h5f.create_dataset('A', data=train_data['A'])
                h5f.create_dataset('B', data=train_data['B'])
                h5f.close()
        else:
            test_data = load_images(os.path.join(param.image_dir, "test"), n_images=param.testing_image)


        print('Building model')
        model = deblur_model(param)


        if param.is_train:
            print('Training model')
            model.train(train_data, 
                            batch_size=param.batch_size, 
                            pre_trained_model=param.model_name, 
                            save_freq = param.save_freq,
                            epoch_num = param.epoch_num,
                            generate_image_freq = param.generate_image_freq)
        else:
            print('Debluring')
            model.generate(test_data, batch_size=param.batch_size, trained_model=param.model_name)
