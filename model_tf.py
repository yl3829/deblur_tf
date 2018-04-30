import tensorflow as tf
import numpy as np
from utils import res_block, PSNR, deprocess_image
import time
import os


class deblur_model():
    def __init__(self, 
                 param,
                 LAMBDA_A = 100
                 ):
        # input:
        #   d_param(dict): parameters need for discriminator 
        #   g_param(dict): parameters need for generator
        self.param = param
        self.LAMBDA_A = LAMBDA_A
        self.d_merge = []
        self.g_merge = []
        self.generator_model()
        self.discriminator_model()
        self.init_loss()
        
    
    def generator_model(self):
        # built the generator model 
        input_size = self.param.g_input_size
        ngf = self.param.ngf
        n_downsampling =  self.param.n_downsampling
        output_nc = self.param.output_nc
        n_blocks_gen = self.param.n_blocks_gen

        with tf.variable_scope('g_model'):
            self.real_A = tf.placeholder(dtype=tf.float32, shape=[None,None,None,3], name='real_A')
            g_input = self.real_A

            _out = tf.pad( g_input, [ [0, 0], [3, 3], [3, 3], [0, 0] ], mode="REFLECT" )
            _out = tf.layers.conv2d(_out, filters=ngf, kernel_size=(7,7), strides=(1,1), padding='VALID')
            _out = tf.layers.batch_normalization(_out)
            _out = tf.nn.relu(features=_out)

            for i in range(n_downsampling):
                mult = 2**i
                _out = tf.pad(_out,[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
                _out = tf.layers.conv2d(_out, filters=ngf*mult*2, kernel_size=(3, 3), strides=(2, 2), padding='VALID')
                _out = tf.layers.batch_normalization(_out)
                _out = tf.nn.relu(features=_out)

            mult = 2**n_downsampling
            for i in range(n_blocks_gen):
                _out = res_block(_out, ngf*mult, use_dropout=True)

            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                #_out = tf.pad(_out,[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
                _out = tf.layers.conv2d_transpose(_out, filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=(2, 2), padding='SAME')
                #_out = tf.pad(_out,[[0,0],[1,0],[1,0],[0,0]], mode="CONSTANT")
                _out = tf.layers.batch_normalization(_out)
                _out = tf.nn.relu(features=_out)

            _out = tf.pad( _out, [ [0, 0], [3, 3], [3, 3], [0, 0] ], mode="REFLECT" )
            _out = tf.layers.conv2d(_out, filters=output_nc, kernel_size=(7,7), strides=(1,1), padding='VALID')
            _out = tf.tanh(x=_out)

            _out = tf.add(_out, g_input)

            _out = tf.clip_by_value( _out, clip_value_min = -1, clip_value_max = 1 )

            self.fake_B = _out


        # output
        #    self.fake_B
        # self.real_A = # a place for the input of blury image
        # # implementations
        # # ...
        
        # self.fake_B = # a tensor for the output of the generator_model
        # pass
    
    def discriminator_model(self):
        # take input from self.d_fake_B and self.real_B
        # output the result of discrrminator
        input_size = self.param.d_input_size
        ndf = self.param.ndf
        kernel_size = self.param.kernel_size
        n_layers = self.param.n_layers_D
        
        
        with tf.variable_scope('d_model'):
            alpha = tf.random_uniform([1])
            
            self.d_fake_B = tf.placeholder(dtype=tf.float32, shape=[None,input_size,input_size,3], name='d_fake_b')
            self.real_B = tf.placeholder(tf.float32, shape=[None,input_size,input_size,3], name='real_B') # a placeholder for the real sharp image
            self.interpolates = alpha * self.real_B + (1-alpha) * self.d_fake_B
            
            d_input = tf.concat([self.d_fake_B,self.real_B,self.interpolates],0)
            bs = tf.shape(self.d_fake_B)[0]
            
            _out = tf.layers.conv2d(d_input, filters=ndf, kernel_size=kernel_size, strides=(2,2), name='conv0',padding='same')
            _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            for n in range(1,n_layers):
                nf_mulf = min(2**n, 8)
                
                _out = tf.layers.conv2d(_out, filters=ndf*nf_mulf, kernel_size=kernel_size, strides=(2,2), name='conv{}'.format(n), padding='same')
                _out = tf.layers.batch_normalization(_out)
                _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            nf_mulf = min(2**n_layers, 8)
            _out = tf.layers.conv2d(_out, filters=ndf*nf_mulf, kernel_size=kernel_size, strides=(1,1), name='conv{}'.format(n_layers), padding='same')
            _out = tf.layers.batch_normalization(_out)
            _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            self.fake_D = _out[:bs]
            self.real_D = _out[bs:2*bs]
            self.disc_interpolates = _out[2*bs:]
            
            
        with tf.variable_scope('d_model', reuse=True):
                        
            d_input = self.fake_B
            
            _out = tf.layers.conv2d(d_input, filters=ndf, kernel_size=kernel_size, strides=(2,2), name='conv0',padding='same', reuse=True)
            _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            for n in range(1,n_layers):
                nf_mulf = min(2**n, 8)
                
                _out = tf.layers.conv2d(_out, filters=ndf*nf_mulf, kernel_size=kernel_size, strides=(2,2), name='conv{}'.format(n), padding='same', reuse=True)
                _out = tf.layers.batch_normalization(_out)
                _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            nf_mulf = min(2**n_layers, 8)
            _out = tf.layers.conv2d(_out, filters=ndf*nf_mulf, kernel_size=kernel_size, strides=(1,1), name='conv{}'.format(n_layers), padding='same', reuse=True)
            _out = tf.layers.batch_normalization(_out)
            _out = tf.nn.leaky_relu(features=_out, alpha=0.2)
            
            self.g_fake_D = _out


    
    def wgangp_loss(self,LAMBDA=10):
        # input:
        #    fake_D: a tensor generated by d model used generated images
        #    real_D: a tensor generated by d model used real sharp images
        # return:
        #    d_loss: loss for discriminator
        #    g_gan_loss: loss for generator
        self.g_gan_loss = -tf.reduce_mean(self.g_fake_D)
        self.g_merge.append(tf.summary.scalar('generator_wgan_loss', self.g_gan_loss))
        
        grad = tf.gradients(self.disc_interpolates,self.interpolates)
        gradient_penalty = tf.reduce_mean((tf.norm(grad, ord=2, axis=1)-1) ** 2) * LAMBDA
        self.d_loss = tf.reduce_mean(self.fake_D) - tf.reduce_mean(self.real_D) + gradient_penalty
        self.d_merge.append(tf.summary.scalar('discriminator_loss', self.d_loss))
        
    
    def preceptual_loss(self):
        # input:
        #    fake_B: a tensor generated by generator_model
        #    real_B: a place holder for the real sharp image 
        # return:
        #    p_loss: the preceptual loss for generator
        _in = tf.concat([self.fake_B,self.real_B],axis=0)
        bs = tf.shape(self.fake_B)[0]
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256,256,3), input_tensor=_in)
        vgg.trainable = False
        _out = vgg.get_layer('block3_conv3').output
        self.p_loss = tf.losses.mean_squared_error(_out[bs:],_out[:bs])
        self.g_merge.append(tf.summary.scalar('generator_preceptual_loss', self.p_loss))
    
    def init_loss(self):
        # combine the loss of g model and d model 
        # and apply them to two different optimizer.
        self.wgangp_loss()
        self.preceptual_loss()
        self.g_loss = self.LAMBDA_A*self.g_gan_loss + self.p_loss
        self.g_merge.append(tf.summary.scalar('generator_loss', self.g_loss))
        # get the variables in discriminator and generator
        tvars = tf.trainable_variables()
        
        d_vars = [var for var in tvars if 'd_model' in var.name]
        g_vars = [var for var in tvars if 'g_model' in var.name]
        
        self.D_trainer = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
        self.G_trainer = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)
    
    def train(self, 
              train_data,
              batch_size = 16,
              epoch_num = 10,
              critic_updates=5,
              save_freq = 2,
              val_freq = 200,
              show_freq = 1,
              pre_trained_model=None):
        # implement training on two models
        cur_model_name = 'Deblur_{}'.format(int(time.time()))
        sharp, blur = train_data['B'], train_data['A']
        min_loss = np.inf
        i = 0
        
        with tf.Session() as sess:
            merge_D = tf.summary.merge(self.d_merge)
            merge_G = tf.summary.merge(self.g_merge)
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                    #writer = tf.summary.FileWriterCache.get('log/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass
            
            for epoch in range(epoch_num):
                permutated_indexes = np.random.permutation(sharp.shape[0])
                
                for index in range(int(blur.shape[0] / batch_size)):
                    batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
                    
                    sharp_batch = sharp[batch_indexes]
                    blur_batch = blur[batch_indexes]
                    #print('------------------------------------')
                    generated_images  = sess.run(self.fake_B, feed_dict={self.real_A: blur_batch})
                    #print('------------------------------------')
                    for _ in range(critic_updates):
                        d_loss, _, d_merge_result = sess.run([self.d_loss,self.D_trainer, merge_D],
                                                           feed_dict={self.real_B: sharp_batch, self.d_fake_B: generated_images})
                    writer.add_summary(d_merge_result, i) 
                    
                    g_loss, _, g_merge_result = sess.run([self.g_loss,self.G_trainer, merge_G],
                                                       feed_dict={self.real_A: blur_batch, self.real_B: sharp_batch})
                    
                    writer.add_summary(g_merge_result, i)
                    if (i+1) % show_freq == 0:
                        print("{}/{} batch in {}/{} epochs, discriminator loss: {}, generator loss: {}".format(index+1,
                                                                                                               int(blur.shape[0] / batch_size),
                                                                                                               epoch+1,
                                                                                                               epoch_num,
                                                                                                               d_loss,
                                                                                                               g_loss))
                    if (i+1) % save_freq == 0:
                        if not os.path.exists('model/'):
                            os.makedirs('model/')
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                        print('{} Saved'.format(cur_model_name))
                    i += 1
        
    
    def generate(self,test_data,trained_model):
        # generate deblured image
        y_test, x_test = test_data['B'], test_data['A']
        size=x_test.shape[0]
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
        
            saver.restore(sess, '{}'.format(trained_model))
            print("Model restored.")
            
            ##Generate deblurred images
            generated_test = sess.run(self.fake_B, feed_dict={self.real_A: x_test})
            generated = np.array([deprocess_image(img) for img in generated_test])
            x_test = deprocess_image(x_test)
            y_test = deprocess_image(y_test)
            
            ##save image
            for i in range(generated_test.shape[0]):
                y = y_test[i, :, :, :]
                x = x_test[i, :, :, :]
                img = generated[i, :, :, :]
                output = np.concatenate((y, x, img), axis=1)
                im = Image.fromarray(output.astype(np.uint8))
                im.save('deblur'+str(i))
            
            ##Calculate Peak Signal Noise Ratio(PSNR)
            psnr=0
            for i in range(generated_test.shape[0]):
                    y = y_test[i, :, :, :]
                    img = generated[i, :, :, :]
                    psnr = psnr+PSNR(y,img)
            psnr_mean = psnr/size
            print("PSNR of testing data: "+str(psnr_mean))
           
                