# Load pickled data
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
from conv_utils import *
import argparse

parser = argparse.ArgumentParser(description='Advanced Lane Finding')
parser.add_argument('--training_data_path', help='Path the trianing data for Classifier', dest='class_path', type=str, default='./training_data')

args = parser.parse_args()

# Retrieve training data
cars = glob.glob('{}/vehicles/*/*.png'.format(args.class_path))
notcars = glob.glob('{}/non-vehicles/*/*.png'.format(args.class_path))

X_cars = []
y_cars = []
X_notcars = []
y_notcars = []

for car in tqdm(cars):
    X_cars.append(mpimg.imread(car))

for notcar in tqdm(notcars):
    X_notcars.append(mpimg.imread(notcar))

X = np.vstack((X_cars, X_notcars)).astype(np.float64)

y = np.hstack((np.ones(len(X_cars)), np.zeros(len(X_notcars)))).astype(np.uint8)

# Create Train, Validate and Test sets
X_train, X_vt, y_train, y_vt = train_test_split(X, y, train_size=0.6, test_size=0.4, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_vt, y_vt, train_size=0.5, test_size=0.5, shuffle=True)

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]
n_train_y = y_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# Set classes to two
n_classes = 2

# Print some info
print("Number of training examples =", n_train)
print("Number of training labels =", len(y_train))
print("Number of validation examples = ", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

X_train_gray_normalized = []
y_train_gray_normalized = []
X_valid_gray_normalized = []
X_test_gray_normalized = []

num_additional_images = 0
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
for i in tqdm(range(n_train)):
    X_train_gray_normalized.append(convert_to_gray_and_normalize(X_train[i]))
    y_train_gray_normalized.append(y_train[i])
    for j in range(0,num_additional_images):
        rotation = random.choice(list(chain(range(-30,-5), range(5,30))))
        scale = random.uniform(0.75, 1.0)
        X_train_gray_normalized.append(normalize_image(rotate_image(convert_to_gray(X_train[i]), rotation, scale)))
        y_train_gray_normalized.append(y_train[i])

for i in tqdm(range(n_validation)):
    X_valid_gray_normalized.append(convert_to_gray_and_normalize(X_valid[i]))

for i in tqdm(range(n_test)):
    X_test_gray_normalized.append(convert_to_gray_and_normalize(X_test[i]))

X_train_gray_normalized = np.array(X_train_gray_normalized)
X_valid_gray_normalized = np.array(X_valid_gray_normalized)
X_test_gray_normalized = np.array(X_test_gray_normalized)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 64,64,1), name="INPUT")
y = tf.placeholder(tf.uint8, (None), name="LABELS")
one_hot_y = tf.one_hot(y,n_classes)
keep_prob = tf.placeholder(tf.float32, name="KEEP_PROB")

EPOCHS = 30
#EPOCHS = 1
BATCH_SIZE = 128
rate = 0.001

logits = TrafficSignNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = (num_additional_images+1)*n_train
    
    previous_accuracy = 0.0
    print("Training...")
    print()
    for i in range(EPOCHS):
        print("EPOCH {} ...".format(i+1))
        X_train_gray_normalized, y_train_gray_normalized = shuffle(X_train_gray_normalized, y_train_gray_normalized)
        for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray_normalized[offset:end], y_train_gray_normalized[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.4})

        validation_accuracy = evaluate(X_valid_gray_normalized, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        if (validation_accuracy >= previous_accuracy):
            saver.save(sess, './vehicle_detect')
            print("Model saved")
            previous_accuracy = validation_accuracy
        print()

### Test model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray_normalized, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

### Classify new image
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    img = mpimg.imread('./test_images/test1.jpg')
    res_img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
    gray_norm_img = convert_to_gray_and_normalize(res_img)

    my_image_X = np.array([gray_norm_img])


    my_image_prediction = sess.run(tf.argmax(logits, 1), feed_dict={x:my_image_X, keep_prob: 1.0})
    print(logits.name)
    print("My image prediction = ", my_image_prediction)
    

