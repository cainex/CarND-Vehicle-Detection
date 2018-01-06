import cv2
import matplotlib.pyplot as plt
import argparse
import glob
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Lane Finding')
    parser.add_argument('--training_data_path', help='Path the trianing data for Classifier', dest='class_path', type=str, default='./training_data')
    args = parser.parse_args()

    cars = glob.glob('{}/vehicles/*/*.png'.format(args.class_path))
    notcars = glob.glob('{}/non-vehicles/*/*.png'.format(args.class_path))

    cars_imgs = []
    notcars_imgs = []
    for i in range(8):
        cars_imgs.append(cv2.imread(cars[i*1000]))
        notcars_imgs.append(cv2.imread(notcars[i*1000]))

    fig, ax = plt.subplots(ncols=4, nrows=4)
    ax[0][0].imshow(cars_imgs[0])
    ax[0][1].imshow(cars_imgs[1])
    ax[0][2].imshow(cars_imgs[2])
    ax[0][3].imshow(cars_imgs[3])
    ax[1][0].imshow(cars_imgs[4])
    ax[1][1].imshow(cars_imgs[5])
    ax[1][2].imshow(cars_imgs[6])
    ax[1][3].imshow(cars_imgs[7])
    ax[0][0].set_title('car')
    ax[0][1].set_title('car')
    ax[0][2].set_title('car')
    ax[0][3].set_title('car')
    ax[1][0].set_title('car')
    ax[1][1].set_title('car')
    ax[1][2].set_title('car')
    ax[1][3].set_title('car')

    ax[2][0].imshow(notcars_imgs[0])
    ax[2][1].imshow(notcars_imgs[1])
    ax[2][2].imshow(notcars_imgs[2])
    ax[2][3].imshow(notcars_imgs[3])
    ax[3][0].imshow(notcars_imgs[4])
    ax[3][1].imshow(notcars_imgs[5])
    ax[3][2].imshow(notcars_imgs[6])
    ax[3][3].imshow(notcars_imgs[7])
    ax[2][0].set_title('notcar')
    ax[2][1].set_title('notcar')
    ax[2][2].set_title('notcar')
    ax[2][3].set_title('notcar')
    ax[3][0].set_title('notcar')
    ax[3][1].set_title('notcar')
    ax[3][2].set_title('notcar')
    ax[3][3].set_title('notcar')

    plt.tight_layout()
    plt.savefig('output_images/dataset_sample.jpg')

    params = get_default_classifier_parameters()
    car_features, car_feature_img = get_hog_features(cars_imgs[0][:,:,2], params['orient'], params['pix_per_cell'], params['cell_per_block'], vis=True)
    notcar_features, notcar_feature_img = get_hog_features(notcars_imgs[0][:,:,2], params['orient'], params['pix_per_cell'], params['cell_per_block'], vis=True)

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0][0].imshow(cars_imgs[0])
    ax[0][1].imshow(car_feature_img, cmap='gray')
    ax[0][0].set_title('car')
    ax[0][1].set_title('car HOG features')

    ax[1][0].imshow(notcars_imgs[0])
    ax[1][1].imshow(notcar_feature_img, cmap='gray')
    ax[1][0].set_title('notcar')
    ax[1][1].set_title('notcar HOG features')
    plt.tight_layout()
    plt.savefig('output_images/hog_images.jpg')

    test_images = []

    for i in range(1,7):
        test_images.append(cv2.cvtColor(cv2.imread('output_images/test{}_image.jpg'.format(i)), cv2.COLOR_BGR2RGB))

    fig, ax = plt.subplots(ncols=2, nrows=3)
    ax[0][0].imshow(test_images[0])
    ax[0][1].imshow(test_images[1])
    ax[1][0].imshow(test_images[2])
    ax[1][1].imshow(test_images[3])
    ax[2][0].imshow(test_images[4])
    ax[2][1].imshow(test_images[5])
    plt.tight_layout()
    plt.savefig('output_images/test_images.jpg')

        