import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt

import cPickle

import urllib2

slim = tf.contrib.slim

import vgg
import vgg_preprocessing
from vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

from load_synthia import load_synthia
import utils

class DSN(object):
    
    def __init__(self, seq_name, fc7_size, exp_folder, exp_subfolder, no_classes=14):
	
	
	self.exp_dir = os.path.join(exp_folder, seq_name, exp_subfolder)
	self.fc7_size = fc7_size
	
	self.seq_name = seq_name
	self.no_classes = no_classes
	self.log_dir = os.path.join(self.exp_dir,'logs')
	self.vgg_checkpoint_path = './vgg_16.ckpt'
	self.scale = 1
	
    def vgg_encoding(self, processed_images, is_training, reuse=False): 
		
	with slim.arg_scope(vgg.vgg_arg_scope()):

	    fc7, pool4, pool3 = vgg.vgg_16(processed_images,
                                           num_classes=self.no_classes,
					   is_training=is_training,
					   spatial_squeeze=False,
					   fc_conv_padding='SAME',
					   fc7_size=self.fc7_size,
					   return_fc7 = True)
				
	    self.fc7 = fc7
	    self.pool4 = pool4
	    self.pool3 = pool3
	    return fc7, pool4, pool3
    
    def semantic_extractor_32s(self, fc7):
	
	#~ with tf.device('/gpu:1'):
	
	with tf.variable_scope('semantic_extractor'):
	    
	    upsample_factor = 32
	    
	    upsample_filter_np = utils.bilinear_upsample_weights(upsample_factor,
								 no_input_channels = self.fc7_size,
								 no_output_channels = self.no_classes)

	    upsample_filter_tensor = tf.Variable(upsample_filter_np)

	    downsampled_logits_shape = tf.shape(fc7)

	    upsampled_logits_shape = tf.pack([
					      downsampled_logits_shape[0],
					      downsampled_logits_shape[1] * upsample_factor,
					      downsampled_logits_shape[2] * upsample_factor,
					      self.no_classes
					     ])

	    upsampled_logits = tf.nn.conv2d_transpose(fc7, upsample_filter_tensor, output_shape=upsampled_logits_shape, strides=[1, upsample_factor, upsample_factor, 1])
	    
		
	    self.logits = tf.sigmoid(upsampled_logits)
	    return self.logits
	    
    def semantic_extractor_16s(self, fc7, pool4):
	
	#~ with tf.device('/gpu:1'):
	
	with tf.variable_scope('semantic_extractor'):
	    

	    upsample_factor_fc7 = 2      # fc7        shape x2, e.g. from 7x7 to 14x14
	    upsample_factor_pool4 = 1    # pool4      shape x1, e.g. from 14x14 to 14x14 
	    upsample_factor_all = 16     # fc7__pool4 shape x16, e.g. from 14x14 to 224x224
	    
	    fc7_shape = tf.shape(fc7)
	    pool4_shape = tf.shape(pool4)
	    
	    first_upsample_filter_np_fc7 = utils.bilinear_upsample_weights(upsample_factor_fc7, no_input_channels = self.fc7_size, no_output_channels = self.no_classes)
	    first_upsample_filter_tensor_fc7 = tf.Variable(first_upsample_filter_np_fc7)
	    first_upsampled_fc7_shape = tf.pack([fc7_shape[0], fc7_shape[1] * upsample_factor_fc7, fc7_shape[2] * upsample_factor_fc7, self.no_classes])
	    first_upsampled_fc7 = tf.nn.conv2d_transpose(fc7, first_upsample_filter_tensor_fc7, output_shape=first_upsampled_fc7_shape, strides=[1, upsample_factor_fc7, upsample_factor_fc7, 1])

	    first_filter_np_pool4 = utils.bilinear_upsample_weights(1, no_input_channels = 512, no_output_channels = self.no_classes)
	    first_filter_tensor_pool4 = tf.Variable(first_filter_np_pool4)
	    first_pool4_shape = tf.pack([pool4_shape[0], pool4_shape[1] * upsample_factor_pool4, pool4_shape[2] * upsample_factor_pool4, self.no_classes])
	    first_pool4 = tf.nn.conv2d_transpose(pool4, first_filter_tensor_pool4, output_shape=first_pool4_shape, strides=[1, upsample_factor_pool4, upsample_factor_pool4, 1])

	    fc7_pool4 = tf.concat(3, (first_upsampled_fc7, first_pool4)) #(1x14x14x13) concat (1x14x14x13) -> (1x14x14x26) NOT SURE ABOUT THIS!
	    fc7_pool4_shape = tf.shape(fc7_pool4)
	    
	    output_shape = tf.pack([fc7_pool4_shape[0], fc7_pool4_shape[1] * upsample_factor_all, fc7_pool4_shape[2] * upsample_factor_all, self.no_classes])
	    
    	    last_upsample_filter_np = utils.bilinear_upsample_weights(upsample_factor_all, no_input_channels = self.no_classes * 2, no_output_channels = self.no_classes)
	    last_upsample_filter_tensor = tf.Variable(last_upsample_filter_np)

	    output_logits = tf.nn.conv2d_transpose(fc7_pool4, last_upsample_filter_tensor, output_shape=output_shape, strides=[1, upsample_factor_all, upsample_factor_all, 1])
		
	    #~ self.logits = tf.sigmoid(output_logits)
	    return output_logits

    def semantic_extractor_8s(self, fc7, pool4, pool3):
	
	#~ with tf.device('/gpu:1'):
	
	with tf.variable_scope('semantic_extractor'):
	    

	    upsample_factor_fc7 = 4      # fc7             shape x4, e.g. from 7x7 to 28x28
	    upsample_factor_pool4 = 2    # pool4           shape x2, e.g. from 14x14 to 28x28 
	    upsample_factor_pool3 = 1    # pool3           shape x1, e.g. from 28x28 to 28x28
	    upsample_factor_all = 8      # fc7_pool4_pool3 shape x8, e.g. from 28x28 to 224x224
	    
	    fc7_shape = tf.shape(fc7)
	    pool4_shape = tf.shape(pool4)
	    pool3_shape = tf.shape(pool3)
	    
	    first_upsample_filter_np_fc7 = utils.bilinear_upsample_weights(upsample_factor_fc7, no_input_channels = self.fc7_size, no_output_channels = self.no_classes)
	    first_upsample_filter_tensor_fc7 = tf.Variable(first_upsample_filter_np_fc7)
	    first_upsampled_fc7_shape = tf.pack([fc7_shape[0], fc7_shape[1] * upsample_factor_fc7, fc7_shape[2] * upsample_factor_fc7, self.no_classes])
	    first_upsampled_fc7 = tf.nn.conv2d_transpose(fc7, first_upsample_filter_tensor_fc7, output_shape=first_upsampled_fc7_shape, strides=[1, upsample_factor_fc7, upsample_factor_fc7, 1])

	    first_filter_np_pool4 = utils.bilinear_upsample_weights(1, no_input_channels = 512, no_output_channels = self.no_classes)
	    first_filter_tensor_pool4 = tf.Variable(first_filter_np_pool4)
	    first_pool4_shape = tf.pack([pool4_shape[0], pool4_shape[1] * upsample_factor_pool4, pool4_shape[2] * upsample_factor_pool4, self.no_classes])
	    first_pool4 = tf.nn.conv2d_transpose(pool4, first_filter_tensor_pool4, output_shape=first_pool4_shape, strides=[1, upsample_factor_pool4, upsample_factor_pool4, 1])

	    first_filter_np_pool3 = utils.bilinear_upsample_weights(1, no_input_channels = 256, no_output_channels = self.no_classes)
	    first_filter_tensor_pool3 = tf.Variable(first_filter_np_pool3)
	    first_pool3_shape = tf.pack([pool3_shape[0], pool3_shape[1] * upsample_factor_pool3, pool3_shape[2] * upsample_factor_pool3, self.no_classes])
	    first_pool3 = tf.nn.conv2d_transpose(pool3, first_filter_tensor_pool3, output_shape=first_pool3_shape, strides=[1, upsample_factor_pool3, upsample_factor_pool3, 1])

	    fc7_pool4_pool3 = tf.concat(3, (first_upsampled_fc7, first_pool4, first_pool3)) #(1x28x28x14) concat (1x28x28x14) concat (1x28x28x14) -> (1x28x28x42) NOT SURE ABOUT THIS!
	    fc7_pool4_pool3_shape = tf.shape(fc7_pool4_pool3)
	    
	    output_shape = tf.pack([fc7_pool4_pool3_shape[0], fc7_pool4_pool3_shape[1] * upsample_factor_all, fc7_pool4_pool3_shape[2] * upsample_factor_all, self.no_classes])
	    
    	    last_upsample_filter_np = utils.bilinear_upsample_weights(upsample_factor_all, no_input_channels = self.no_classes * 3, no_output_channels = self.no_classes)
	    last_upsample_filter_tensor = tf.Variable(last_upsample_filter_np)

	    output_logits = tf.nn.conv2d_transpose(fc7_pool4_pool3, last_upsample_filter_tensor, output_shape=output_shape, strides=[1, upsample_factor_all, upsample_factor_all, 1])
		
	    return output_logits	
    
    def feature_generator(self, noise, reuse=False, is_training=True):
    
	'''
	Takes in input noise, and generates 
	f_z, which is handled by the net as 
	f(x) was handled.  
	'''
    
	with tf.variable_scope('feature_generator', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):
		
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
				    activation_fn=tf.nn.relu, is_training=is_training):
		    
		    net = slim.fully_connected(noise, 1024, activation_fn = tf.nn.relu, scope='sgen_fc1')
		    net = slim.batch_norm(net, scope='sgen_bn1')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc2')
		    net = slim.batch_norm(net, scope='sgen_bn2')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc3')
		    net = slim.batch_norm(net, scope='sgen_bn3')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, self.fc7_size, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
	
    def feature_discriminator(self, inputs, reuse=False, is_training=True):

	with tf.variable_scope('feature_discriminator',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
				    activation_fn=tf.nn.relu, is_training=is_training):
		    if self.mode=='train_feature_generator':
			net = slim.fully_connected(inputs, 512, activation_fn = utils.lrelu, scope='sdisc_fc1')
			#~ net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc2')
		    
		    elif self.mode=='train_domain_invariant_encoder':
			net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sdisc_2_fc1')
			net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='sdisc_2_fc2')
			net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='sdisc_2_fc3')
			#~ net = slim.fully_connected(net, 2048, activation_fn = utils.lrelu, scope='sdisc_2_fc4')
		
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_2_prob')
		    return net
	    
    def build_model(self, mode='pretrain'):
	
	self.mode=mode
	
	if self.mode=='train_semantic_extractor':
	
	    self.images = tf.placeholder(tf.float32, [None, 224 * self.scale,224 * self.scale, 3], 'images')
	    self.annotations = tf.placeholder(tf.float32, [None, 224 * self.scale,224 * self.scale, 1], 'annotations')
	    self.is_training = tf.placeholder(tf.bool)

	    labels_tensors = [tf.to_float(tf.equal(self.annotations, i)) for i in range(self.no_classes)]
	    self.labels_tensors = labels_tensors
	    
	    combined_mask = tf.concat(3,labels_tensors)
	    
		
	    flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, self.no_classes))

	    image_float = tf.to_float(self.images, name='ToFloat')
	    processed_images = tf.subtract(image_float, tf.constant([_R_MEAN, _G_MEAN, _B_MEAN]))
	    
	    # extracting VGG-16 representation, up to the (N-1) layer
	    
	    self.vgg_output, pool4, pool3 = self.vgg_encoding(processed_images, self.is_training)
	    self.vgg_output_flat = tf.squeeze(self.vgg_output)
	    
	    vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])
	    vgg_except_fc8_weights = slim.get_variables_to_restore(exclude= ['vgg_16/fc7','vgg_16/fc8'])
	    
	    # extracting semantic representation
	    
	    #~ logits = self.semantic_extractor_32s(self.vgg_output)
	    #~ logits = self.semantic_extractor_16s(self.vgg_output, pool4)
	    logits = self.semantic_extractor_8s(self.vgg_output, pool4, pool3)
	    
	    flat_logits = tf.reshape(tensor=logits, shape=(-1, self.no_classes))
	    
	    self.flat_logits = flat_logits
	    self.flat_labels = flat_labels

	    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
								      labels=flat_labels)
    
	    self.cross_entropy_sum = tf.reduce_sum(cross_entropies) 

	    self.pred = tf.argmax(logits, dimension=3)
	    
	    self.pred_tensors = [tf.to_float(tf.equal(self.pred, i)) for i in range(self.no_classes)]

	    self.probabilities = tf.nn.softmax(logits)

	    # Optimizers

	    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)

	    # no re-training of VGG-16 variables
	    
	    print '\n\n\n\n\n\nWARNING - TRAINNIG ALL VGG-16 PARAMETERS\n\n\n\n\n\n'

	    t_vars = tf.trainable_variables()
	    self.train_vars = t_vars#[var for var in t_vars if ('vgg_16' not in var.name) or ('fc6' in var.name) or ('fc7' in var.name) or ('fc8' in var.name)]

	    # train op
	    with tf.variable_scope('training_op',reuse=False):
		self.train_op = slim.learning.create_train_op(self.cross_entropy_sum, optimizer, variables_to_train=self.train_vars)

	    tf.summary.scalar('cross_entropy_loss', self.cross_entropy_sum)

	    self.merged_summary_op = tf.summary.merge_all()


	    # necessary to load VGG-16 weights

	    self.read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
		self.vgg_checkpoint_path,
		vgg_except_fc8_weights)

	    self.vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)
	    	
	if self.mode=='train_feature_generator':
	
	    self.fx = tf.placeholder(tf.float32, [None, self.fc7_size], 'images')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
			
	    self.fzy = self.feature_generator(self.noise, is_training=True) 

	    self.logits_real = self.feature_discriminator(self.fx, reuse=False) 
	    self.logits_fake = self.feature_discriminator(self.fzy, reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    self.d_optimizer = tf.train.AdamOptimizer(0.00001)
	    self.g_optimizer = tf.train.AdamOptimizer(0.0001)
	    
	    t_vars = tf.trainable_variables()
	    d_vars = [var for var in t_vars if 'feature_discriminator' in var.name]
	    g_vars = [var for var in t_vars if 'feature_generator' in var.name]
	    
	    self.g_vars = g_vars
	    
	    
	    # train op
	    with tf.variable_scope('source_train_op',reuse=False):
		self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
		self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)
	    
	    # summary op
	    d_loss_summary = tf.summary.scalar('feature_discriminator_loss', self.d_loss)
	    g_loss_summary = tf.summary.scalar('feature_generator_loss', self.g_loss)
	    self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

	    for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	    	
	if self.mode=='train_domain_invariant_encoder':
	
            self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'mnist_images')
	    self.is_training = tf.placeholder(tf.bool)
	    
	    self.images = tf.concat(0, [self.src_images, self.trg_images])
	    
	    self.images = tf.to_float(self.images, name='ToFloat')
	    self.images = tf.subtract(self.images, tf.constant([_R_MEAN, _G_MEAN, _B_MEAN]))
	    
	    
	    self.fx = self.vgg_encoding(self.images, self.is_training)
	    
	    vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])
	    vgg_except_fc8_weights = slim.get_variables_to_restore(exclude= ['vgg_16/fc7','vgg_16/fc8'])
	    
	    self.fzy = self.feature_generator(self.noise) 
	    
	    
	    # E losses
	    
	    self.logits_E_real = self.feature_discriminator(self.fzy)
	    
	    self.logits_E_fake = self.feature_discriminator(self.fx, reuse=True)
	    
	    self.DE_loss_real = tf.reduce_mean(tf.square(self.logits_E_real - tf.ones_like(self.logits_E_real)))
	    self.DE_loss_fake = tf.reduce_mean(tf.square(self.logits_E_fake - tf.zeros_like(self.logits_E_fake)))
	    
	    self.DE_loss = self.DE_loss_real + self.DE_loss_fake 
	    
	    self.E_loss = tf.reduce_mean(tf.square(self.logits_E_fake - tf.ones_like(self.logits_E_fake)))
	    	    
	    # Optimizers
	    
            self.DE_optimizer = tf.train.AdamOptimizer(0.0000001)
            self.E_optimizer = tf.train.AdamOptimizer(0.0000001)
            
            
            t_vars = tf.trainable_variables()
            self.E_vars = [var for var in t_vars if ('vgg' in var.name)]
            self.DE_vars = [var for var in t_vars if 'feature_discriminator' in var.name]
            
            # train op
	    try:
		with tf.variable_scope('training_op',reuse=False):
		    self.E_train_op = slim.learning.create_train_op(self.E_loss, self.E_optimizer, variables_to_train=self.E_vars)
		    self.DE_train_op = slim.learning.create_train_op(self.DE_loss, self.DE_optimizer, variables_to_train=self.DE_vars)
		    
	    except:
		with tf.variable_scope('training_op',reuse=True):
		    self.E_train_op = slim.learning.create_train_op(self.E_loss, self.E_optimizer, variables_to_train=E_vars)
		    self.DE_train_op = slim.learning.create_train_op(self.DE_loss, self.DE_optimizer, variables_to_train=DE_vars)
		    
	    
            
            # summary op
            E_loss_summary = tf.summary.scalar('E_loss', self.E_loss)
            DE_loss_summary = tf.summary.scalar('DE_loss', self.DE_loss)
            self.summary_op = tf.summary.merge([E_loss_summary, DE_loss_summary])
            

            for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
		
    	    # necessary to load VGG-16 weights

	    self.read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
		self.vgg_checkpoint_path,
		vgg_except_fc8_weights)

	    self.vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)


####################################################################################################################################################################################


    def train_semantic_extractor(self):
		
	def computeIoU(y_pred_batch, y_true_batch):
	    qwe = np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]) 
	    return np.mean(qwe[qwe>-1])
	    
	def pixelAccuracy(y_pred, y_true):
	    y_pred = np.argmax(y_pred,axis=0)
	    y_true = np.argmax(y_true,axis=0)
	    y_pred = y_pred * (y_true>0)

	    if np.sum(y_true>0) > 0.:
		return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)	
	    else:
		return -1
		
	self.build_model('train_semantic_extractor')

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	#~ config = tf.ConfigProto(device_count = {'GPU': 0})
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	

	with tf.Session(config=config) as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
     
	    #~ # Run the initializers.
   	    #~ print '\n\n\n\n\n\nWARNING - LOADING ALL VGG-16 PARAMETERS\n\n\n\n\n\n'

	    #~ sess.run(tf.global_variables_initializer())
	    #~ self.read_vgg_weights_except_fc8_func(sess)
	    #~ sess.run(self.vgg_fc8_weights_initializer)
	    #~ variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, os.path.join(self.exp_dir,'model/segm_model'))
	     
	    saver = tf.train.Saver(self.train_vars)
	    
	    images, annotations = load_synthia(self.seq_name, no_elements=900)
	    
	    feed_dict = {self.images: images,
			 self.annotations: annotations,
			 self.is_training: False}

	    EPOCHS = 1000000
	    BATCH_SIZE = 1
	    
	    
	    
	    for e in range(0, EPOCHS):
		
		losses = []
		mIoUs = []
		
		print e
		#~ print 'Saving model.'
		#~ saver.save(sess, os.path.join(self.exp_dir,'model/segm_model'))
		
		for n, start, end in zip(range(len(images)), range(0,len(images),BATCH_SIZE), range(BATCH_SIZE,len(images),BATCH_SIZE)):
		    
		    if n % 100 == 0:
			print n
			    
		    feed_dict = {self.images: images[start:end], self.annotations: annotations[start:end], self.is_training: True}

		    fc7, flat_logits, flat_labels = sess.run([self.fc7, self.flat_logits, self.flat_labels], feed_dict=feed_dict)
		    loss, summary_string, pred_tensors, labels_tensors = sess.run([self.cross_entropy_sum, self.merged_summary_op, self.pred_tensors, self.labels_tensors], feed_dict=feed_dict)
		    
		    labels_tensors = np.squeeze(np.array(labels_tensors))
		    pred_tensors = np.squeeze(np.array(pred_tensors))

		    

		    sess.run(self.train_op, feed_dict=feed_dict)

		    summary_string_writer.add_summary(summary_string, e)

		    
		    losses.append(loss)
		    mIoUs.append(computeIoU(pred_tensors,labels_tensors))
		    
		    
		    
		print 'mIoU:',np.mean(np.array(mIoUs))
		print e,'- current average loss:',str(np.array(losses).mean())
		pred_np, probabilities_np = sess.run([self.pred, self.probabilities], feed_dict={self.images: images[0:1], self.annotations: annotations[0:1], self.is_training: False})
		plt.imsave(self.exp_dir+'/images/'+str(e)+'.png', np.squeeze(pred_np))	    
		plt.imsave(self.exp_dir+'/images/'+str(e)+'_image.png', np.squeeze(images[0:1]))	    
		plt.imsave(self.exp_dir+'/images/'+str(e)+'_annotation.png', np.squeeze(annotations[0:1]))	    
	    
		#~ pred_np, probabilities_np = sess.run([self.pred, self.probabilities], feed_dict={self.images: images[6:7], self.annotations: annotations[6:7], self.is_training: False})
		#~ plt.imsave('./experiments/'+self.seq_name+'/images/'+str(e)+'_2.png', np.squeeze(pred_np))	    
		#~ plt.imsave('./experiments/'+self.seq_name+'/images/'+str(e)+'_image_2.png', np.squeeze(images[6:7]))	    
		#~ plt.imsave('./experiments/'+self.seq_name+'/images/'+str(e)+'_annotation_2.png', np.squeeze(images[6:7]))	    
		
		
		

	    summary_string_writer.close()
	    
    def train_feature_generator(self):
	
	epochs=10000
	batch_size=64
	noise_dim=100

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})

	source_images, source_annotations = load_synthia(self.seq_name, no_elements=900)
		
	source_features = self.extract_VGG16_features(source_images)
	
	self.build_model(mode='train_feature_generator')
	
        with tf.Session() as sess:
	
	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver(slim.get_model_variables(scope='feature_generator'))
	    
            tf.global_variables_initializer().run()
	    
	    t = 0
	    
	    for i in range(epochs):
		 
		#~ print 'Epoch',str(i), '.....................................................................'
		
		for start, end in zip(range(0, len(source_images), batch_size), range(batch_size, len(source_images), batch_size)):
			
		    if t % 5000 == 0:  
			saver.save(sess, os.path.join(self.exp_dir,'model/sampler'))
 	    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {self.noise: Z_samples, self.fx: source_features[start:end]}
	    
		    avg_D_fake = sess.run(self.logits_fake, feed_dict)
		    avg_D_real = sess.run(self.logits_real, feed_dict)
		    
		    sess.run(self.d_train_op, feed_dict)
		    sess.run(self.g_train_op, feed_dict)
		    
		    if (t+1) % 200 == 0:
			summary, dl, gl = sess.run([self.summary_op, self.d_loss, self.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, gl))
			print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
			
    def train_domain_invariant_encoder(self, seq_2_name):
	
	print 'Adapting from ' + self.seq_name + ' to ' + seq_2_name
	
	epochs=1000000
	batch_size=8
	noise_dim=100
		
	self.build_model('train_domain_invariant_encoder')

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})

        with tf.Session() as sess:
	    
	    print 'Loading weights.'

	    # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, os.path.join(self.exp_dir,'model/segm_model'))
	    restorer.restore(sess, self.exp_dir+'/model/segm_model')
	    
	    variables_to_restore = slim.get_model_variables(scope='feature_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, os.path.join(self.exp_dir+'model/sampler'))
	    restorer.restore(sess, self.exp_dir+'/model/sampler')

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver(self.E_vars)

	    source_images, source_annotations = load_synthia(self.seq_name, no_elements=900)
	    target_images, target_annotations = load_synthia(seq_2_name, no_elements=900)
	    
	    print ('Start training.')
	    trg_count = 0
	    t = 0
	    
	    for step in range(100000):

		#~ if t % 1000 == 0:
		    #~ print 'Saving model.'  
		    #~ saver.save(sess, self.exp_dir+'/model/di_encoder_new')
		
		trg_count += 1
		t+=1
		
		i = step % int(source_images.shape[0] / batch_size)
		j = step % int(target_images.shape[0] / batch_size)
		
		src_images = source_images[i*batch_size:(i+1)*batch_size]
		trg_images = target_images[j*batch_size:(j+1)*batch_size]
		noise = utils.sample_Z(batch_size,100,'uniform')
		
		feed_dict = {self.src_images: src_images, self.trg_images: trg_images, self.noise: noise, self.is_training: True}
		
		sess.run(self.E_train_op, feed_dict) 
		sess.run(self.DE_train_op, feed_dict) 
		
		logits_E_real,logits_E_fake = sess.run([self.logits_E_real,self.logits_E_fake],feed_dict) 
		 
		if (step+1) % 100 == 0:
		    
		    summary, E, DE = sess.run([self.summary_op, self.E_loss, self.DE_loss], feed_dict)
		    summary_writer.add_summary(summary, step)
		    print ('Step: [%d] E: [%.3f] DE: [%.3f] E_real: [%.2f] E_fake: [%.2f]' \
			       %(step+1, E, DE, logits_E_real.mean(),logits_E_fake.mean()))
	    
    def eval_semantic_extractor(self, seq_2_name, train_stage='pretrain'):
	
	
	def computeIoU(y_pred_batch, y_true_batch):
	    qwe = np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]) 
	    return np.mean(qwe[qwe>-1])
	    
	    
	def pixelAccuracy(y_pred, y_true):
	    y_pred = np.argmax(y_pred,axis=0)
	    y_true = np.argmax(y_true,axis=0)
	    y_pred = y_pred * (y_true>0)

	    if np.sum(y_true>0) > 0.:
		return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)	
	    else:
		return -1
		
	self.build_model('train_semantic_extractor')

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})


	source_images, source_annotations = load_synthia(self.seq_name, no_elements=10)
	source_features = np.zeros((len(source_images),self.fc7_size))
	source_losses = np.zeros((len(source_images), 1))
	source_preds = np.zeros((len(source_images),224,224))
	source_mIoU = np.zeros((len(source_images),))
	
	target_images, target_annotations = load_synthia(seq_2_name, no_elements=10)
	target_features = np.zeros((len(target_images),self.fc7_size))
	target_preds = np.zeros((len(target_images),224,224))
	target_losses = np.zeros((len(target_images), 1))
	target_mIoU = np.zeros((len(target_images),))

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.exp_dir+'/model/segm_model')
	    
	    if train_stage=='dsn':
		print 'Loading adapted fc6-fc7 weights.'
		variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name)]
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.exp_dir+'/model/di_encoder')
		
	    
	    saver = tf.train.Saver(self.train_vars)

	    print 'Evaluating SOURCE - ' + self.seq_name
	    
	    
	    
	    for n, image, annotation in zip(range(len(source_images)), source_images, source_annotations):
		
		if n%100==0:
		    print n 
		
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.expand_dims(annotation,0), self.is_training: False}
		feat, pred, loss, pred_tensors, labels_tensors = sess.run([self.vgg_output_flat, self.pred, self.cross_entropy_sum, self.pred_tensors, self.labels_tensors], feed_dict=feed_dict)
		
		labels_tensors = np.squeeze(np.array(labels_tensors))
		pred_tensors = np.squeeze(np.array(pred_tensors))
		

		source_mIoU[n] = computeIoU(pred_tensors,labels_tensors)
		source_features[n] = feat
		source_preds[n] = pred
		source_losses[n] = loss
	    
	    print 'Average source loss: ' + str(source_losses.mean())
	    print 'Average source mIoU: ' + str(np.mean(source_mIoU))
	    
	    print 'Evaluating TARGET - ' + seq_2_name
	    
	    for n, image, annotation in zip(range(len(target_images)), target_images, target_annotations):
		
		if n%100==0:
		    print n 
		
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.expand_dims(annotation,0), self.is_training: False}
		feat, pred, loss, pred_tensors, labels_tensors = sess.run([self.vgg_output_flat, self.pred, self.cross_entropy_sum, self.pred_tensors, self.labels_tensors], feed_dict=feed_dict)
		
		labels_tensors = np.squeeze(np.array(labels_tensors))
		pred_tensors = np.squeeze(np.array(pred_tensors))
		
		
		target_mIoU[n] = computeIoU(pred_tensors,labels_tensors)
		target_features[n] = feat
		target_preds[n] = pred
		target_losses[n] = loss
		
	    print 'Average target loss: ' + str(target_losses.mean())
	    print 'Average source mIoU: ' + str(np.mean(target_mIoU))
	    
	    print 'break'
	    
    def extract_VGG16_features(self, source_images, train_stage='pretrain'):
	
	print 'Extracting VGG_16 features.'
	
	self.build_model(mode='train_semantic_extractor')
	
	source_features = np.zeros((len(source_images),self.fc7_size))

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, os.path.join(self.exp_dir,'model/segm_model'))
	    	    	    
	    if train_stage=='dsn':
		print 'Loading adapted fc6-fc7 weights.'
		variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name)]
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, os.path.join(self.exp_dir,'model/di_encoder'))

	    print 'Extracting VGG-16 features.'
	    
	    for n, image in enumerate(source_images):
		
		if n%100==0:
		    print n 
		    
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.zeros((1,224,224,1)), self.is_training: False}
		feat = sess.run(self.vgg_output_flat, feed_dict=feed_dict)
		source_features[n] = feat
	    
	    return source_features
		    
    def features_to_pkl(self, seq_2_names = ['...'], train_stage='dsn'):
	
	source_images, _ = load_synthia(self.seq_name, no_elements=900)
		
	source_features = self.extract_VGG16_features(source_images, train_stage=train_stage)
	tf.reset_default_graph()
	
	target_features = dict()
	
	for s in seq_2_names:
	    target_images, _ = load_synthia(s, no_elements=900)
	    target_features[s] = self.extract_VGG16_features(target_images, train_stage=train_stage)
	    tf.reset_default_graph()
	
	self.build_model(mode='train_feature_generator')
	
        with tf.Session() as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
	    
	    print ('Loading feature generator.')
	    variables_to_restore = slim.get_model_variables(scope='feature_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.exp_dir+'/model/sampler')
	    
	    n_samples = 900
            noise = utils.sample_Z(n_samples,100,'uniform')
	    
	    feed_dict = {self.noise: noise, self.fx: source_features[1:2]}
	    
	    fzy = sess.run([self.fzy], feed_dict)
	    
	    with open(self.exp_dir+'/features_'+train_stage+'.pkl','w') as f:
		cPickle.dump((source_features, target_features, fzy), f, cPickle.HIGHEST_PROTOCOL)
    
    def extract_all_maps(self, train_stage='pretrain'):
	

	
	self.build_model('train_semantic_extractor')

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})


	source_images, source_annotations = load_synthia('SYNTHIA-SEQS-01-NIGHT', no_elements=900)
		     
	source_features = np.zeros((len(source_images),self.fc7_size))
	source_losses = np.zeros((len(source_images), 1))
	
	source_preds = np.zeros((len(source_images),224,224))
	

	with tf.Session(config = config) as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.exp_dir+'/model/segm_model')
	    
	    if train_stage=='dsn':
		print 'Loading adapted fc6-fc7 weights.'
		variables_to_restore = [i for i in slim.get_model_variables() if ('fc6' in i.name) or ('fc7' in i.name)]
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.exp_dir+'/model/di_encoder_new')
		
	    
	    saver = tf.train.Saver(self.train_vars)

	    print 'Evaluating SOURCE - ' + self.seq_name
	    
	    for n, image, annotation in zip(range(len(source_images)), source_images, source_annotations):
		
		if n%1==0:
		    print n
		     
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.expand_dims(annotation,0), self.is_training: False}
		feat, pred, loss, pred_tensors, labels_tensors = sess.run([self.vgg_output_flat, self.pred, self.cross_entropy_sum, self.pred_tensors, self.labels_tensors], feed_dict=feed_dict)
		
		labels_tensors = np.squeeze(np.array(labels_tensors))
		pred_tensors = np.squeeze(np.array(pred_tensors))
		

		plt.imsave(self.exp_dir+'/images/whole_dataset_NIGHT/'+str(n)+'.png', np.squeeze(pred))	    
		plt.imsave(self.exp_dir+'/images/whole_dataset_NIGHT/'+str(n)+'_image.png', np.squeeze(image))	    
		plt.imsave(self.exp_dir+'/images/whole_dataset_NIGHT/'+str(n)+'_annotation.png', np.squeeze(annotation))	    

    def mean_IoU(predictions, annotations):
	return 0
	
	
####################################################################################################################################################################################


if __name__ == "__main__":
    
    
    GPU_ID = sys.argv[1]
    MODE = 'train_semantic_extractor'
    SEQ_NAME = 'SYNTHIA-SEQS-01-NIGHT'
    FC7_SIZE = int(sys.argv[2])
    EXP_FOLDER = '/cvgl2/u/rvolpi/experiments/'
    EXP_SUBFOLDER = str(sys.argv[2])

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    
    model = DSN(seq_name=SEQ_NAME, fc7_size=FC7_SIZE, exp_folder = EXP_FOLDER, exp_subfolder = EXP_SUBFOLDER)
	    
    if MODE == 'train_semantic_extractor':    
	print 'Training Semantic Extractor.'
	model.train_semantic_extractor()
	tf.reset_default_graph()

    elif MODE == 'train_feature_generator':
	print 'Training Feature Generator'
	model.train_feature_generator()
	tf.reset_default_graph()
    
    elif MODE == 'train_domain_invariant_encoder':
	print 'Training Domain-Invariant Encoder.'
	model.train_domain_invariant_encoder(seq_2_name='SYNTHIA-SEQS-01-NIGHT')
	tf.reset_default_graph()

    elif MODE == 'save_features':
	print 'Saving Features.'
	seq_2_names = ['SYNTHIA-SEQS-01-NIGHT']
	model.features_to_pkl(seq_2_names = seq_2_names, train_stage='dsn')
	
    elif MODE == 'evaluate_semantic_extractor':        
	print 'Evaluate Semantic Extractor.'
	model.eval_semantic_extractor(seq_2_name='SYNTHIA-SEQS-01-NIGHT', train_stage='dsn')
    
    elif MODE == 'extract_all_maps':
	print 'Extracting All Maps.'
	model.extract_all_maps(train_stage='pretrain')
	
    else:
	raise Exception('Unrecognized mode.')
	    
	    

	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	
	    

#TO PLOT IMAGES





#~ cmap = plt.get_cmap('bwr')

#~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#~ ax1.imshow(np.uint8(pred_np.squeeze() != 1), vmax=1.5, vmin=-0.4, cmap=cmap)
#~ ax1.set_title('Argmax. Iteration # ' + str(i))
#~ probability_graph = ax2.imshow(probabilities_np.squeeze()[:, :, 0])
#~ ax2.set_title('Probability of the Class. Iteration # ' + str(i))

#~ plt.colorbar(probability_graph)
#~ plt.show()






#~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#~ 
#~ cmap = plt.get_cmap('bwr')
#~ 
#~ ax1.imshow(np.uint8(final_predictions.squeeze() != 1),
	   #~ vmax=1.5,
	   #~ vmin=-0.4,
	   #~ cmap=cmap)
#~ 
#~ ax1.set_title('Final Argmax')
#~ 
#~ probability_graph = ax2.imshow(final_probabilities.squeeze()[:, :, 0])
#~ ax2.set_title('Final Probability of the Class')
#~ plt.colorbar(probability_graph)
#~ 
#~ plt.show()
