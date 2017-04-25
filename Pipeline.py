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

s_x = [293,1143,535,750]
s_y = [711,711,493,493]
    
d_x = [313,1030,313,1030]

d_y = [711,711,375,375]



    
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
        gray = image
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
        gray = image
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
        gray = image
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return dir_binary

    
    
    
    def calcGradients(self,image, absx,absy,mag,abs_dir):
        
        
#        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = image[:,:,0]
        
        
        
        
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        abs_x =  self.abs_sobel_thresh( blur, 'x', absx[0], absx[1])
        abs_y =  self.abs_sobel_thresh( blur, 'y', absy[0], absy[1])
        mag_x =  self.mag_thresh(blur, mag[0], mag[1])
        dir_x =  self.dir_threshold(blur, abs_dir[0], abs_dir[1])
         
        return abs_x, abs_y, mag_x, dir_x
         
    
    
    
    def colorThr(self,image,s_thresh):
        # Convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Get the s channel as its most robust
        s_channel = hls[:,:,2]


        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        return s_binary
    
    
    def getBinaryImg (self, a,b,c,d,s_binary ):
        
        final = np.zeros_like(a)
        final[((a == 1) & (b == 1)) | ((c == 1) & (d == 1))] = 1
    
    
        binary = np.zeros_like(final)
        binary[(s_binary == 1) | (final == 1)] = 1
        
        
        return binary
    

           
    def getWarpedImg(self,imge, combined_binary):
        
        
        img_size = (imge.shape[1], imge.shape[0])
        #    

        
        src = np.int32([ [s_x[0],s_y[0] ], [ s_x[1],s_y[1]], 
                                         [s_x[3],s_y[3]], 
                                         [s_x[2],s_y[2]]])
        
        #    dst = np.float32([[313,711], [993,711], 
        #                                     [313,375], 
        #                                     [993,375]])
        dst = np.int32([ [d_x[0],d_y[0] ], [ d_x[1],d_y[1]], 
                                         [d_x[3],d_y[3]], 
                                         [d_x[2],d_y[2]]])
        
        
         
        cv2.polylines(imge,np.int32([src]),True,(255,0,0), 5)
        #    cv2.polylines(combined_binary,np.int32([src]),True,(255,0,0), 5)
        
        
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
            # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective( combined_binary, M, img_size)
        

        return warped
    
    def fitLine(self, warped , leftx, lefty,rightx ,righty):
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        
        
        
        return left_fitx, right_fitx, ploty
    
    
    
        
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
            img_mask = source +'frame*.jpg'     # create a mask using glob
            
        images = self.collectCalibImages(img_mask)
        
        return images
    
    
    




   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    