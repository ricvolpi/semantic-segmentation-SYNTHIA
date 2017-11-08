import numpy as np
import numpy.random as npr
import tensorflow as tf


def sample_Z(m, n, mode='uniform'):
	if mode=='uniform':
		return npr.uniform(-1., 1., size=[m, n])
	if mode=='gaussian':
		return np.clip(npr.normal(0,0.1,(m,n)),-1,1)

def lrelu(input, leak=0.2, scope='lrelu'):
    
    return tf.maximum(input, leak*input)

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, no_input_channels, no_output_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        no_output_channels,
                        no_input_channels), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(no_output_channels):
	for j in xrange(no_input_channels):
	    weights[:, :, i, j] = upsample_kernel

    rnd_perturbation = tf.random_normal(shape=[filter_size,filter_size,no_output_channels,no_input_channels], mean=0.0, stddev=0.001, dtype=tf.float32)

    weights = weights #+ rnd_perturbation 
    return weights





