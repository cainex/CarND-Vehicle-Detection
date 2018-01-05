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

# TODO: Fill this in based on where you saved the training and testing data

#training_file = "./data/train.p"
# validation_file= "./data/valid.p"
# testing_file = "./data/test.p"

#with open(training_file, mode='rb') as f:
#    train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

#print(train['labels'])
# X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

cars = glob.glob('/mnt/raid/projects/udacity/sdc_nd/datasets/vehicle-detection/vehicles/*/*.png')
notcars = glob.glob('/mnt/raid/projects/udacity/sdc_nd/datasets/vehicle-detection//non-vehicles/*/*.png')
#notcars += glob.glob('/mnt/raid/projects/udacity/sdc_nd/datasets/vehicle-detection//non-vehicles/*/*.jpg')

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

X_train, X_vt, y_train, y_vt = train_test_split(X, y, train_size=0.6, test_size=0.4, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_vt, y_vt, train_size=0.5, test_size=0.5, shuffle=True)

print(y_train)

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

# TODO: How many unique classes/labels there are in the dataset.
# sign_labels = []
# labelreader = csv.reader(open('signnames.csv'), delimiter=",")
# for next_label in labelreader:
# #    print(next_label)
#     sign_labels.append(next_label)
# sign_labels.pop(0)

# sign_labels_lookup = [''] * np.array(sign_labels).shape[0]
# for next_label in sign_labels:
#     sign_labels_lookup[int(next_label[0])]=next_label[1]

#print(sign_labels_lookup)

n_classes = 2

print("Number of training examples =", n_train)
print("Number of training labels =", len(y_train))
print("Number of validation examples = ", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
#%matplotlib inline

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16,16))

# ax[0][0].hist(train['labels'], bins=n_classes)
# ax[0][0].set_title('Dist of signs in training set')

# ax[0][1].hist(valid['labels'], bins=n_classes)
# ax[0][1].set_title('Dist of signs in validation set')

# ax[0][2].hist(test['labels'], bins=n_classes)
# ax[0][2].set_title('Dist of signs in test set')

ax[1][0].imshow(X_train[0])
ax[1][0].set_title(y_train[0])
ax[1][1].imshow(X_train[1000])
ax[1][1].set_title(y_train[1000])
ax[1][2].imshow(X_train[2000])
ax[1][2].set_title(y_train[2000])

ax[2][0].imshow(rgb2gray(X_train[0]), cmap='gray')
ax[2][1].imshow(rgb2gray(X_train[1000]), cmap='gray')
ax[2][2].imshow(rgb2gray(X_train[2000]), cmap='gray')

# plt.show()
# plt.cla()
# plt.clf()
# plt.close()

X_train_gray_normalized = []
y_train_gray_normalized = []
X_valid_gray_normalized = []
X_test_gray_normalized = []

def convert_to_gray(x):
#    gray_image = img_as_ubyte(exposure.equalize_adapthist(x))
#    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
#    return np.reshape((gray_image-[128])/[128], [32,32,1])  
#    return np.reshape((cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)-[128])/[128], [32,32,1])  
#    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return rgb2gray(x)

def rotate_image(x, rot, scale):
    M = cv2.getRotationMatrix2D((16,16),rot,scale)
    return cv2.warpAffine(x,M,(64,64),borderMode=cv2.BORDER_REPLICATE)
      

def normalize_image(x):
    return np.reshape(x/[255], [64,64,1])

def convert_to_gray_and_normalize(x):
    return normalize_image(convert_to_gray(x))


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

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

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

def TrafficSignGrayNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # Layer 1
    l1_conv = conv_layer(x, (5,5), 1, 32)
    l1_1_1_conv = conv_layer(l1_conv, (1,1), 32, 18)

    l1_ksize = [1,2,2,1]
    l1_pool_strides = [1,4,4,1]
    l1_1_1_conv = tf.nn.max_pool(l1_1_1_conv,l1_ksize,l1_pool_strides,"VALID")

    # Layer 2
    l2_conv = conv_layer(l1_1_1_conv, (5,5), 18, 48)
    l2_1_1_conv = conv_layer(l2_conv, (1,1), 48, 16)

    l2_ksize = [1,2,2,1]
    l2_pool_strides = [1,2,2,1]
    l2_1_1_conv = tf.nn.max_pool(l2_1_1_conv,l2_ksize,l2_pool_strides,"VALID")

    l2_flat = tf.reshape(l2_1_1_conv, [-1,400])
    
    # Layer 3
    l3_dense = tf.layers.dense(l2_flat, 300)
    l3_dense = tf.nn.relu(l3_dense)
    l3_dense = tf.nn.dropout(l3_dense, keep_prob)

    # Layer 4
    l4_dense = tf.layers.dense(l3_dense, 200)
    l4_dense = tf.nn.relu(l4_dense)
    l4_dense = tf.nn.dropout(l4_dense, keep_prob)

    # Layer 5
    l5_dense = tf.layers.dense(l4_dense, 128)
    l5_dense = tf.nn.relu(l5_dense)
    l5_dense = tf.nn.dropout(l5_dense, keep_prob)

    # Output Layer
    logits = tf.layers.dense(l5_dense, 2, name = "output_layer")
    
    return logits

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

#logits = TrafficSignGrayNet(x, keep_prob)
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
            saver.save(sess, './traffic_sign')
            print("Model saved")
            previous_accuracy = validation_accuracy
        print()
        
    #saver.save(sess, './traffic_sign')
    #print("Model saved")

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
    

