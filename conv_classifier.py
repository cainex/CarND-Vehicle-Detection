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
    def __init__(self):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('traffic_sign.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        self.graph = tf.get_default_graph()
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
#        for n in tf.get_default_graph().as_graph_def().node:
#            print(n.name)
        self.x = self.graph.get_tensor_by_name("INPUT:0")
        self.keep_prob = self.graph.get_tensor_by_name("KEEP_PROB:0")
        self.logits = self.graph.get_tensor_by_name("output_layer/BiasAdd:0")
        

    def classify_images(self, images):
        train_images = []
        for image in images:
            # res_img = cv2.resize(images[i], (64,64), interpolation=cv2.INTER_CUBIC)
            # gray_norm_img = self.convert_to_gray_and_normalize(res_img)
            gray_norm_img = self.convert_to_gray_and_normalize(image)
            train_images.append(gray_norm_img)

        x_batch = np.array(train_images)
#        result = self.sess.run(tf.argmax(self.logits,1), feed_dict={self.x:x_batch, self.keep_prob: 1.0} )
        results = self.sess.run(tf.nn.softmax(self.logits), feed_dict={self.x:x_batch, self.keep_prob: 1.0} )
        return_result = []
        for result in results:
            if result[1] > 0.75:
                return_result.append(1)
            else:
                return_result.append(0)

        return(return_result)



    def __exit__(self):
        self.sess.close()

    def convert_to_gray(self, x):
    #    gray_image = img_as_ubyte(exposure.equalize_adapthist(x))
    #    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    #    return np.reshape((gray_image-[128])/[128], [32,32,1])  
    #    return np.reshape((cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)-[128])/[128], [32,32,1])  
    #    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return rgb2gray(x)

    def normalize_image(self, x):
        return np.reshape(x/[255], [64,64,1])

    def convert_to_gray_and_normalize(self, x):
        return self.normalize_image(self.convert_to_gray(x))

# if __name__ == "__main__":
#     clf = conv_classifier()
    
#     my_image = mpimg.imread('./test_images/test1.jpg')

#     print(clf.classify_image(my_image))

