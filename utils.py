import tensorflow as tf 
import numpy as np


def res_block(input, filters, use_dropout=False):
    
    padded_input = tf.pad( input, [ [0, 0], [1, 1], [1, 1], [0, 0] ], 'REFLECTION' )
    
    _out = tf.layers.conv2d(padded_input, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding = 'VALID')
    _out = tf.layers.batch_normalization(_out)
    _out = tf.nn.relu(_out)
    if use_dropout:
        _out = tf.nn.dropout( _out, keep_prob = 0.5 )
    
    _out = tf.pad( _out, [ [0, 0], [1, 1], [1, 1], [0, 0] ], 'REFLECTION' )
    _out = tf.layers.conv2d(input, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding = 'VALID')
    _out = tf.layers.batch_normalization(_out)
    
    result = tf.add( _out, input )
    
    return result




