# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:07:07 2017

@author: Aneeq Mahmood
@email: aneeq.sdc@gmail.com
"""

"""
Pipeline for image calibration , distortion correction, and other processing
"""
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

calib_image_dir = "camera_cal/"
test_image_dir = "test_images/"

    
class Pipeline():
       
    def __init__(self, a,b):
        """ Initializes the pipeline """
        self.NX = a
        self.NY = b
        self.grid = (a,b)
        print("Class Initilaized")
        
    def cameraCalibMatrices(self,img):
        """
        takes a list of images as input and finds true chessboard corners
        """
        print("Getting to know the corners of images")
        imgp = []
        objp = []
        positives = 0
        objponints = np.zeros((self.NX * self.NY,3),np.float32)
        
        objponints[:,:2] = np.mgrid[ 0:self.NX , 0:self.NY].T.reshape(-1,2)
        
        for ind, row in enumerate(img):
        
            #First convert image to grascale
#            gray = cv2.cvtColor(row,cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor( row, cv2.COLOR_RGB2GRAY )

            ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
           
            if (ret == True):
                objp.append(objponints)
                imgp.append(corners)
                positives += 1
         
        
        print("Correctly find corners on", positives,"images out of a total of", len(img)) 
           
           # getting claibration matrices
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None)
        return mtx , dist
        
        
        
    def cameraDistremove(self,img,mtx,dst,val):
         
         """
         use mtx, and dst arrays from cameraCalibMatrices to correct distortion
         in an image 
         """
        
         udst = cv2.undistort(img, mtx, dst, None, mtx)
         if (val):
             return udst
        
         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
         f.tight_layout()
         ax1.imshow(img)
         ax1.set_title('Original Image', fontsize=50)
         ax2.imshow(udst)
         ax2.set_title('Undistorted Image', fontsize=50)
         plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        
         return udst
        
   
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            	abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        abs_binary = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        abs_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[-1])] = 1
        return abs_binary
    
    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary= np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
        return mag_binary
    
    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return dir_binary


         
        
    def collectCalibImages(self,mask):
        """
        Collects all images which are needed to calibrate the camera
        Input. Addresses of all the images in a directory
        Output: All jpeg images in that dir
    
        """
        images = []         # a place to save all camera images
        
        img_names = glob(mask)                   # Collect image names   
         
        for fn in img_names:
        
            img = cv2.imread(fn)
#            images.append(img)
            images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))   # change colorspace and save in images
        
        return images
        
        
    def collectImages(self,source):
        
        """
        Wrapper Function to read all images in a directory
        input: which directory to read form
        """
        if (source == calib_image_dir):
            img_mask = source +'calibration*.jpg'     # create a mask using glob
     
        elif (source == test_image_dir):
            img_mask = source +'*.jpg'     # create a mask using glob
            
        images = self.collectCalibImages(img_mask)
        
        return images