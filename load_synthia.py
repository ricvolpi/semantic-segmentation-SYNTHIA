import numpy as np 
import numpy.random as npr
import pandas as pd
from PIL import Image
from scipy import misc

import cv2

import imageio

import os
import glob

def load_synthia(seq_name, no_elements=1000):
    
    print 'Loading ' + seq_name

    data_dir = './data'
    seq_num = '01'  
    mode = 'DAWN'

    #~ img_dir = os.path.join(data_dir,'SYNTHIA-SEQS-'+seq_num+'-'+mode,'RGB/Stereo_Left/Omni_F')

    #~ img_dir = './data/SYNTHIA/Omni_F_RGB'
    #~ gt_labels_dir = './data/SYNTHIA/Omni_F_GT_LABELS'
    
    img_dir = '/cvgl/group/Synthia/'+seq_name+'/RGB/Stereo_Left/Omni_F'
    gt_labels_dir = '/cvgl/group/Synthia/'+seq_name+'/GT/LABELS/Stereo_Left/Omni_F' 

    img_files = sorted(glob.glob(img_dir+'/*'))[:10]
    gt_labels_files = sorted(glob.glob(gt_labels_dir+'/*'))[:10]

    scale = 1

    images = np.zeros((len(img_files), 736, 1280, 3))
    gt_labels = np.zeros((len(gt_labels_files), 736, 1280))

    for n, img, gt_lab in zip(range(len(img_files)), img_files, gt_labels_files):
	
	#~ print n
	
	img = misc.imread(img)
	img = cv2.resize(img, (1280, 736))
	
	gt_lab = np.asarray(imageio.imread(gt_lab, format='PNG-FI'))[:,:,0]  # uint16
	gt_lab = cv2.resize(gt_lab, (1280, 736), interpolation = cv2.INTER_NEAREST)
	
	
	images[n] = img
	gt_labels[n] = gt_lab
    
    gt_labels[gt_labels==15] = 13
    
    npr.seed(231)
    
    rnd_indices = np.arange(0,len(images))
    npr.shuffle(rnd_indices)
    
    images = images[rnd_indices]
    gt_labels = gt_labels[rnd_indices]
    
    return images, np.expand_dims(gt_labels,3).astype(int)

if __name__=='__main__':
    
    images, gt_labels = load_synthia(seq_name='SYNTHIA-SEQS-01-DAWN',no_elements=100)
    print 'break'







