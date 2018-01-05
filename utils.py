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

def convert_color(img, conv='BGR2YCrCb'):
    if conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif conv == 'BGR2LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif conv == 'BGR2HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif conv == 'BGR2YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      

def calibrate_camera(images, nx, ny):
    '''
    Calibrate the camera.
    Input is a list of calibration images
    '''

    board_shape = (nx, ny)
    objpoints = []
    imgpoints = []

    # Create object reference points
    objp = np.zeros((board_shape[0]*board_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)

    image_size = None
    
#    for fname in tqdm(images):
    for fname in tqdm(images):
        # Open next calibration image
        img = mpimg.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # set image_size for later use if we haven't already set it
        if image_size == None:
            image_size = gray.shape[1::-1]
            
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_shape, None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            cv2.drawChessboardCorners(img, board_shape, corners, ret)
            cv2.imwrite('output_images/calib_{}'.format(basename(fname)), img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    cam_params = {'mtx' : mtx, 'dist' : dist}

    return cam_params

def undistort_image(img, cam_params):
    '''
    Undistorts an image given the image and camera parameters
    '''
    dst = cv2.undistort(img, cam_params['mtx'], cam_params['dist'], None, cam_params['mtx'])

    return dst


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
    return HOG features and visualization
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    '''
    compute binned color features
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    compute color histogram features 
    NEED TO CHANGE bins_range if reading .png files with mpimg!
    '''

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    '''
    extract features from a single image
    '''
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    extract features from a list of images
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in tqdm(imgs):
#        image = mpimg.imread(file)
        image = cv2.imread(file)
        file_features = single_img_features(image, color_space, spatial_size, hist_bins, orient, 
                                            pix_per_cell, cell_per_block, hog_channel,
                                            spatial_feat, hist_feat, hog_feat)
                    
        features.append(file_features)
    # Return list of feature vectors
    return features
    
# Define a function that 
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    takes an image,
    start and stop positions in both x and y, 
    window size (x and y dimensions),  
    and overlap fraction (for both x and y)
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    draw bounding boxes
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def create_classifier(cars, notcars, params):
    print("Extracting car features...")
    car_features = extract_features(cars, color_space=params['color_space'], 
                            spatial_size=params['spatial_size'], hist_bins=params['hist_bins'], 
                            orient=params['orient'], pix_per_cell=params['pix_per_cell'], 
                            cell_per_block=params['cell_per_block'], 
                            hog_channel=params['hog_channel'], spatial_feat=params['spatial_feat'], 
                            hist_feat=params['hist_feat'], hog_feat=params['hog_feat'])
    print("Extracting non-car features...")
    notcar_features = extract_features(notcars, color_space=params['color_space'], 
                            spatial_size=params['spatial_size'], hist_bins=params['hist_bins'], 
                            orient=params['orient'], pix_per_cell=params['pix_per_cell'], 
                            cell_per_block=params['cell_per_block'], 
                            hog_channel=params['hog_channel'], spatial_feat=params['spatial_feat'], 
                            hist_feat=params['hist_feat'], hog_feat=params['hog_feat'])

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Fit classifier using:',params['orient'],'orientations',params['pix_per_cell'],
        'pixels per cell and', params['cell_per_block'],'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 

    train_params = {'kernel':('linear', 'rbf'), 'C':[1,10]}
    svr = SVC()
    svc = GridSearchCV(svr, train_params)
    #svc = LinearSVC()
    svc.fit(X_train, y_train)
    print("Accuracy: {}".format(svc.score(X_test, y_test)))
    print(svc.best_params_)
    return X_scaler, svc 

def get_default_classifier_parameters():
    params = {}
    params['color_space'] = 'YUV'
    params['orient'] = 11
    params['pix_per_cell'] = 16
    params['cell_per_block'] = 2
    params['hog_channel'] = 'ALL'
    params['spatial_size'] = (16,16)
    params['hist_bins'] = 16
    params['spatial_feat'] = False
    params['hist_feat'] = False
    params['hog_feat'] = True
    params['y_start_stop'] = [400, 719]
    return params

