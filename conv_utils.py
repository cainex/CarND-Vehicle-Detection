import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_ubyte
import cv2
from tqdm import tqdm
import random
from itertools import chain
import glob
import tensorflow as tf


def convert_to_gray(x):
    return rgb2gray(x)

def rotate_image(x, rot, scale):
    M = cv2.getRotationMatrix2D((16,16),rot,scale)
    return cv2.warpAffine(x,M,(64,64),borderMode=cv2.BORDER_REPLICATE)

def normalize_image(x):
    return np.reshape(x/[255], [64,64,1])

def convert_to_gray_and_normalize(x):
    return normalize_image(convert_to_gray(x))

def conv_layer(x, filter_shape, input_depth, filter_depth, pad="VALID", name="conv"):
    mu = 0
    sigma = 0.1
    
    l_weights = tf.Variable(tf.truncated_normal([filter_shape[0],filter_shape[1],input_depth,filter_depth], mu, sigma))
    l_bias = tf.Variable(tf.zeros(filter_depth))
    l_strides = [1,1,1,1]
    l_conv = tf.nn.conv2d(x, l_weights, l_strides, padding=pad, name=name)
    l_conv = tf.nn.bias_add(l_conv, l_bias)

    # Activation.
    l_conv = tf.nn.relu(l_conv)

    return l_conv

def inception_layer(x, input_depth, inception_kernels, name="incep"):
    print("Inception layer:: input_depth:", input_depth, " kernels:", inception_kernels, " output_depth:", 4*inception_kernels)
    l_inc_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_11")

    l_inc_55_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_55_11")
    l_inc_55_conv = conv_layer(l_inc_55_11_conv, (5,5), inception_kernels, inception_kernels, pad="SAME", name=name+"_55")
 
    l_inc_33_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_33_11")
    l_inc_33_conv = conv_layer(l_inc_33_11_conv, (3,3), inception_kernels, inception_kernels, pad="SAME", name=name+"_33")

    l_inc_avg_pool_conv = tf.nn.avg_pool(x, [1,2,2,1], [1,1,1,1],"SAME", name=name+"_avgpool")
    l_inc_avg_pool_11_conv = conv_layer(l_inc_avg_pool_conv, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_avgpool_11")

    l_inc_conv = tf.concat([l_inc_11_conv, l_inc_55_conv, l_inc_33_conv, l_inc_avg_pool_11_conv], axis=3)

    return l_inc_conv
    
def TrafficSignNet(x, keep_prob):    
    h_params = {'l1_kernels': 32,
                'l1_maxpool_ksize': [1,2,2,1],
                'l1_maxpool_strides': [1,4,4,1],
                'l2_kernels': 64,
                'l3_kernels': 8,
                'l4_kernels': 16,
                'l4_maxpool_ksize': [1,2,2,1],
                'l4_maxpool_strides': [1,2,2,1],
                'l5_avgpool_ksize': [1,5,5,1],
                'l5_avgpool_strides': [1,1,1,1]}

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # Layer 1 input: 32x32x1  output: 14x14x['l1_kernels']
    l1_conv = conv_layer(x, (5,5), 1, h_params['l1_kernels'], pad="SAME", name="layer1")
    l1_conv = conv_layer(l1_conv, (5,5), h_params['l1_kernels'], h_params['l1_kernels'], name="layer1_a")

    l1_conv = tf.nn.max_pool(l1_conv,h_params['l1_maxpool_ksize'],h_params['l1_maxpool_strides'],"VALID", name="layer1_maxpool")

   # print(tf.size(l1_conv))

    # Layer 2 input: 14x14x['l1_kernels']  output: 10x10x['l2_kernels']
    l2_conv = conv_layer(l1_conv, (5,5), h_params['l1_kernels'], h_params['l2_kernels'], pad="SAME", name="layer2")
    l2_conv = conv_layer(l2_conv, (5,5), h_params['l2_kernels'], h_params['l2_kernels'], name="layer2_a")

    # Layer 3 Inception Layer input: 10x10x['l2_kernels']  output: 10x10x4*['l3_kernels']
    l3_inc_conv = inception_layer(l2_conv, h_params['l2_kernels'], h_params['l3_kernels'], name="layer3")

    # Layer 4 Inception Layer input: 10x10x4*['l3_kernels'] output: 5x5x4*['l4_kernels']
    l4_inc_conv = inception_layer(l3_inc_conv, 4*h_params['l3_kernels'], h_params['l4_kernels'], name="layer4")
    l4_inc_conv = tf.nn.max_pool(l4_inc_conv, h_params['l4_maxpool_ksize'], h_params['l4_maxpool_strides'],"VALID", name="layer4_maxpool")

    # Layer 5 Global Average Pooling input: 5x5x4*['l4_kernels'] output: 1x1x4*['l4_kernels']
    l5_avgpool = tf.nn.avg_pool(l4_inc_conv, h_params['l5_avgpool_ksize'], h_params['l5_avgpool_strides'],"VALID", name="layer5_avgpool")
    l5_avgpool = tf.nn.dropout(l5_avgpool, keep_prob)

    l5_avgpool_flat = tf.reshape(l5_avgpool, [-1, 4*h_params['l4_kernels']])
    # Layer 6 Dense layer to get final logits
    logits = tf.layers.dense(l5_avgpool_flat, 2, name="output_layer")
    
    return logits
