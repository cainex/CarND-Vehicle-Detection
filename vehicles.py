from utils import *
from scipy.ndimage.measurements import label
from conv_classifier import conv_classifier
import matplotlib.pyplot as plt

class vehicles:
    '''
    Class that preforms the vehicle detection and holds the track of
    all detected vehicles
    '''
    def __init__(self, clf, params, X_scaler):
        # Handle to current image to run detection on
        self.img = None

        # SVM classifier
        self.clf = clf
        self.params = params
        self.X_scaler = X_scaler

        # Detected window history
        self.hist_depth = 10
        self.hist_windows = []

        # Detection heat-map
        self.heat_map = None
        self.prethresh_heat_map = None
        self.threshold = 8
        #self.frame_threshold = 2
        self.labels = None

        # Counter for debug images
        self.img_out_counter = 0

        # Convolutional neural network classifier
        self.conv_clf = conv_classifier()
        self.debug = False
        self.debug_prefix = 'output'
        self.debug_increment = False

    def process_image(self, image):
        '''
        Process a single incoming image
        '''

        # set image to detect vehicles
        self.img = image

        # Create a copy to draw final image
        draw_image = np.copy(self.img)

        # Set desired overlap for windows
        overlap = (0.9, 0.9)

        # Create sliding windows to search for
        windows = []
        windows += slide_window(self.img, x_start_stop=[None, None], y_start_stop=[400, 480],
                                xy_window=(64,64), xy_overlap=overlap)
        windows += slide_window(self.img, x_start_stop=[None, None], y_start_stop=[400, 528],
                                xy_window=(96,96), xy_overlap=overlap)
        windows += slide_window(self.img, x_start_stop=[None, None], y_start_stop=[400, 560],
                                xy_window=(128,128), xy_overlap=overlap)
        windows += slide_window(self.img, x_start_stop=[None, None], y_start_stop=[400, 719],
                                xy_window=(196,196), xy_overlap=overlap)

        # Perform seach in generated windows
        hot_windows = self.search_windows_conv(self.img, windows)

        # Capture windows with detections
        self.add_hot_windows(hot_windows)

        # Create heat-map
        self.calculate_heat_map()
        self.labels = label(self.heat_map)

        # Draw labelled boxes
        # window_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 255), thick=6)                    
        # window_img = draw_labeled_bboxes(window_img, self.labels)
        window_img = draw_labeled_bboxes(draw_image, self.labels)

        if self.debug:
            my_prefix = self.debug_prefix
            if self.debug_increment:
                my_prefix = '{}{}'.format(my_prefix, self.img_out_counter)
            search_windows_img = draw_boxes(self.img, windows, color=(255, 0, 255), thick=1) 
            hot_window_img = draw_boxes(self.img, hot_windows, color=(255, 0, 255), thick=4)                    
            cv2.imwrite('output_images/{}_search_windows.jpg'.format(my_prefix), cv2.cvtColor(search_windows_img, cv2.COLOR_RGB2BGR))
            # TODO: add heatmap plot here instead of writing image
            fig, ax = plt.subplots( nrows=1, ncols=1)
            ax.imshow(self.heat_map, cmap='hot')
            plt.savefig('output_images/{}_heat_map.jpg'.format(my_prefix))
            fig, ax = plt.subplots( nrows=1, ncols=1)
            ax.imshow(self.prethresh_heat_map, cmap='hot')
            plt.savefig('output_images/{}_prethresh_heat_map.jpg'.format(my_prefix))
            fig, ax = plt.subplots( nrows=1, ncols=1)
            ax.imshow(self.labels[0], cmap='gray')
            plt.savefig('output_images/{}_labels.jpg'.format(my_prefix))
            cv2.imwrite('output_images/{}_hot_windows.jpg'.format(my_prefix), cv2.cvtColor(hot_window_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite('output_images/{}_final.jpg'.format(my_prefix), cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))
            if self.debug_increment:
                self.img_out_counter += 1

        return window_img

    def search_windows_conv(self, img, windows):
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        test_images = []
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            test_images.append(test_img)
        
        prediction = self.conv_clf.classify_images(test_images)
        
        for i in range(len(windows)):
            if prediction[i] == 1:
                svm_pred = self.search_window_svm(img, windows[i], self.clf, self.X_scaler, color_space=self.params['color_space'], 
                                                  spatial_size=self.params['spatial_size'], hist_bins=self.params['hist_bins'], 
                                                  orient=self.params['orient'], pix_per_cell=self.params['pix_per_cell'], 
                                                  cell_per_block=self.params['cell_per_block'], 
                                                  hog_channel=self.params['hog_channel'], spatial_feat=self.params['spatial_feat'], 
                                                  hist_feat=self.params['hist_feat'], hog_feat=self.params['hog_feat'])
                if svm_pred == 1:
                    on_windows.append(windows[i])
                #cv2.imwrite('output_images/train_data/image_{}.jpg'.format(self.img_out_counter), test_images[i])
                #self.img_out_counter += 1

        #8) Return windows for positive detections
        return on_windows
        
    def search_windows(self, img, windows, clf, scaler, color_space='RGB', 
                        spatial_size=(32, 32), hist_bins=32, 
                        hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel=0, spatial_feat=True, 
                        hist_feat=True, hog_feat=True):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = single_img_features(test_img, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
    
    def search_window_svm(self, img, window, clf, scaler, color_space='RGB', 
                          spatial_size=(32, 32), hist_bins=32, 
                          hist_range=(0, 256), orient=9, 
                          pix_per_cell=8, cell_per_block=2, 
                          hog_channel=0, spatial_feat=True, 
                          hist_feat=True, hog_feat=True):

        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #8) Return windows for positive detections
        return prediction
    
    def find_cars(self, img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        '''
        Function to find hot windows
        '''
        on_windows = []

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.clf.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                   
        return on_windows

    def add_hot_windows(self, hot_windows):
        self.hist_windows.append(hot_windows)
        if (len(self.hist_windows) > self.hist_depth):
            self.hist_windows.pop(0)

    def calculate_heat_map(self):
        self.heat_map = heat = np.zeros_like(self.img[:,:,0]).astype(np.float)

        for windows in self.hist_windows:
            self.add_heat(windows)

        if self.debug:
            self.prethresh_heat_map = np.copy(self.heat_map)
                
        self.apply_threshold(self.threshold)


    def add_heat(self, bbox_list):
        for box in bbox_list:
            self.heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heat_map[self.heat_map <= threshold] = 0
    