import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt

import cPickle

from ConfigParser import *

import urllib2

slim = tf.contrib.slim

import vgg
import vgg_preprocessing
from vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

from load_synthia import load_synthia
import utils

class DSN(object):
    
    def __init__(self, exp_dir):
	
	self.exp_dir = exp_dir
	self.load_exp_config()
	
    def load_exp_config(self):
	
	
	config = ConfigParser()
        config.read(self.exp_dir + '/exp_configuration')
        
	self.vgg_checkpoint_path = config.get('EXPERIMENT_SETTINGS', 'vgg_checkpoint_path')
	#~ self.vgg_checkpoint_path = './vgg_16.ckpt'
	self.synthia_dataset_path = config.get('EXPERIMENT_SETTINGS', 'synthia_dataset_path')
	self.synthia_seqs_type = config.get('EXPERIMENT_SETTINGS', 'synthia_seqs_type')
	self.synthia_seqs_number = config.get('EXPERIMENT_SETTINGS', 'synthia_seqs_number')
	
	self.data_dir = self.synthia_dataset_path + '/SYNTHIA-SEQS-' + self.synthia_seqs_number + '-' + self.synthia_seqs_type

	self.log_dir = os.path.join(self.exp_dir,'logs')
		
	self.training_epochs = config.getint('MAIN_SETTINGS', 'training_epochs')
	self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
	self.learning_rate = config.getfloat('MAIN_SETTINGS', 'learning_rate')
	
	self.fc7_size = config.getint('MODEL_SETTINGS', 'fc7_size')
	self.no_classes = config.getint('MODEL_SETTINGS', 'no_classes')
	
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
	    
    def build_model(self, mode='pretrain'):
	
	self.mode=mode
	
	if self.mode=='train_semantic_extractor':
	
	    self.images = tf.placeholder(tf.float32, [None, 736, 1280, 3], 'images')
	    self.annotations = tf.placeholder(tf.float32, [None, 736, 1280, 1], 'annotations')
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
	    vgg_except_fc8_weights = slim.get_variables_to_restore(exclude= ['vgg_16/fc8'])
	    
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

	    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

	    # no re-training of VGG-16 variables
	    
	    print '\n\n\n\n\n\nWARNING - TRAINING ALL VGG-16 PARAMETERS\n\n\n\n\n\n'

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
	    
	    images, annotations = load_synthia(self.data_dir)
	    
	    feed_dict = {self.images: images,
			 self.annotations: annotations,
			 self.is_training: False}

	    EPOCHS = self.training_epochs
	    BATCH_SIZE = self.batch_size
	    
	    
	    
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


	source_images, source_annotations = load_synthia(self.data_dir)
	source_features = np.zeros((len(source_images),self.fc7_size))
	source_losses = np.zeros((len(source_images), 1))
	source_preds = np.zeros((len(source_images),224,224))
	source_mIoU = np.zeros((len(source_images),))
	
	target_images, target_annotations = load_synthia(self.data_dir)
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
	
	source_images, _ = load_synthia(self.data_dir)
		
	source_features = self.extract_VGG16_features(source_images, train_stage=train_stage)
	tf.reset_default_graph()
	
	target_features = dict()
	
	for s in seq_2_names:
	    target_images, _ = load_synthia(self.data_dir)
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


	source_images, source_annotations = load_synthia(self.data_dir)
		     
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

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    
    model = DSN(exp_dir = '/cvgl2/u/rvolpi/experiments/SYNTHIA-SEQS-01-DAWN/dummy')
	    
    if MODE == 'train_semantic_extractor':    
	print 'Training Semantic Extractor.'
	model.train_semantic_extractor()
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
