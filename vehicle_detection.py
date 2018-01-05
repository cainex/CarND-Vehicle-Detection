from utils import *
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
from moviepy import editor
from sklearn.externals import joblib
from vehicles import vehicles


if __name__ == "__main__":
    ## Handle command-line arguments
    parser = argparse.ArgumentParser(description='Advanced Lane Finding')
    parser.add_argument('--classifier', help='Classifier to use - if none, new classifier will be trained', dest='classifier', type=str, default=None)
    parser.add_argument('--training_data_path', help='Path the trianing data for Classifier', dest='class_path', type=str, default='./training_data')
    parser.add_argument('--detection_parameters', help='Classifier parameters to use for vehicle detection', dest='class_params', type=str, default=None)

    parser.add_argument('--dump_dir', help="Directory to dump images", dest='dump_dir', type=str, default=None)
    parser.add_argument('--test_image', help="test image to use", dest='test_image', type=str, default=None)
    parser.add_argument('--test_video', help='test video file to use', dest='test_video', type=str, default='project_video.mp4')
    parser.add_argument('--output_video', help='name of output video file', dest='output_video', type=str, default='video_out.mp4')
    parser.add_argument('--subclip_start', help='subclip start', dest='subclip_start', type=int, default=None)
    parser.add_argument('--subclip_end', help='subclip end', dest='subclip_end', type=int, default=None)
    parser.add_argument('--debug', help='display debug info into output', dest='debug', action='store_true', default=False)
    args = parser.parse_args()

    class_params = None
    if args.class_params is None:
        class_params = get_default_classifier_parameters()
    else:
        class_params = pickle.load(open(args.class_params, 'rb'))


    ## Train classifier
    clf = None
    X_scaler = None
    if args.classifier is None:
        print("Getting list of classifier training images...")
        cars = glob.glob('{}/vehicles/*/*.png'.format(args.class_path))
        notcars = glob.glob('{}/non-vehicles/*/*.png'.format(args.class_path))
        print("Found {} cars, and {} non-cars...".format(len(cars), len(notcars)))

        print("Training Classifier...")
        X_scaler, clf = create_classifier(cars, notcars, class_params)
        classifier_data = {}
        classifier_data['clf'] = clf
        classifier_data['X_scaler'] = X_scaler
        print("Saving Classifier...")
        joblib.dump(classifier_data, 'classifier.p')
    else:
        print("Loading classifier from {}...".format(args.classifier))
        classifier_data = joblib.load(args.classifier)
        clf = classifier_data['clf']
        X_scaler = classifier_data['X_scaler']



    current_vehicles = vehicles(clf, class_params, X_scaler)

    if (args.test_image == None):
        print("Processing video file:{}".format(args.test_video))
        # TODO : add code to handle processing of a video 
        if args.subclip_start is None:
            clip1 = editor.VideoFileClip(args.test_video)
        else:
            clip1 = editor.VideoFileClip(args.test_video).subclip(args.subclip_start, args.subclip_end)
        vid_clip = clip1.fl_image(current_vehicles.process_image)
        vid_clip.write_videofile(args.output_video, audio=False)
    else:
        # test undistortion of a calibration image
        print("processing test image...")
 
        test_image = mpimg.imread(args.test_image)

        final_image = current_vehicles.process_image(test_image)

        cv2.imwrite('output_image.jpg', final_image)
