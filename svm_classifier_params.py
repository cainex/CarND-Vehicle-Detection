import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import basename
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from sklearn.externals import joblib



if __name__ == "__main__":
    print("Loading classifier from classifier.p...")
    classifier_data = joblib.load('classifier.p')
    clf = classifier_data['clf']
    X_scaler = classifier_data['X_scaler']
    print(clf.best_params_)
