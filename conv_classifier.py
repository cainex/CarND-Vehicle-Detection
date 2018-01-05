# Load pickled data
import numpy as np
import pandas as pd
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

class conv_classifier:
    '''
    Class to load a saved Conv Network and perform classification on supplied batch of images
    '''
    def __init__(self):
        # tensorflow session
        self.sess = tf.Session()

        # grab saved network
        saver = tf.train.import_meta_graph('vehicle_detect.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("INPUT:0")
        self.keep_prob = self.graph.get_tensor_by_name("KEEP_PROB:0")
        self.logits = self.graph.get_tensor_by_name("output_layer/BiasAdd:0")

    def classify_images(self, images):
        '''
        Perform classification on supplied images. Uses a softmax output. Since
        we only have two classes, we will set a threshold to return a possitive detection
        to hopefully remove some noise from the detection
        '''        
        train_images = []
        for image in images:
            gray_norm_img = self.convert_to_gray_and_normalize(image)
            train_images.append(gray_norm_img)

        x_batch = np.array(train_images)
        results = self.sess.run(tf.nn.softmax(self.logits), feed_dict={self.x:x_batch, self.keep_prob: 1.0} )
        return_result = []
        for result in results:
            if result[1] > 0.75:
                return_result.append(1)
            else:
                return_result.append(0)

        return(return_result)

    def __exit__(self):
        '''
        destructor
        '''
        self.sess.close()

    def convert_to_gray(self, x):
        '''
        Convert image to grayscale
        '''
        return rgb2gray(x)

    def normalize_image(self, x):
        '''
        Normalize image
        '''
        return np.reshape(x/[255], [64,64,1])

    def convert_to_gray_and_normalize(self, x):
        '''
        Convert image to grayscale and normalize for input to Conv Network
        '''
        return self.normalize_image(self.convert_to_gray(x))

# if __name__ == "__main__":
#     clf = conv_classifier()
    
#     my_image = mpimg.imread('./test_images/test1.jpg')

#     print(clf.classify_image(my_image))

