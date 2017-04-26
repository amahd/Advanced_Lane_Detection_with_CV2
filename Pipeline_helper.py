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

src = np.int32([ [s_x[0],s_y[0] ], [ s_x[1],s_y[1]], 
                                         [s_x[3],s_y[3]], 
                                         [s_x[2],s_y[2]]])
        

dst = np.int32([ [d_x[0],d_y[0] ], [ d_x[1],d_y[1]], 
                                         [d_x[3],d_y[3]], 
                                         [d_x[2],d_y[2]]])
src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

    
class Pipeline_helper():
       
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
         
    
    
    
    def colorThr_s(self,image,s_thresh):
        # Convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Get the s channel as its most robust
        s_channel = hls[:,:,2]


        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        return s_binary
    
    
    def colorThr_l(self,image,l_thresh):
        # Convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Get the s channel as its most robust
        l_channel = hls[:,:,1]


        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        
        return l_binary
    
    def colorThr(self,image,s_thresh,l_thresh):
        
        
        s_binary = self.colorThr_s(image,s_thresh)
        l_binary = self.colorThr_s(image,l_thresh)
        
        sl_binary = np.zeros_like(l_binary)
        sl_binary[ ((s_binary == 1) & (l_binary == 1)) ] = 1
        return sl_binary
    
    
    def getBinaryImg (self, a,b,c,d,s_binary ):
        
        final = np.zeros_like(a)
        final[((a == 1) & (b == 1)) | ((c == 1) & (d == 1))] = 1
    
    
        binary = np.zeros_like(final)
        binary[(s_binary == 1) | (final == 1)] = 1
        
        
        return binary
    

           
    def getWarpedImg(self,imge, combined_binary):
        
        
        img_size = (imge.shape[1], imge.shape[0])
          
        img = np.copy(imge)
        cv2.polylines(img,np.int32([src]),True,(255,0,0), 5)       
        
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
            # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective( combined_binary, M, img_size)
        

        return warped, img
    
    def fitCoeff(self,  leftx, lefty,rightx ,righty):
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit


    def fitVector(self, warped ,left_fit,right_fit):
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        return left_fitx, right_fitx, ploty
    
    def getRadiusCurve(self,left_fit, right_fit, ploty):
        
        y_eval = np.float(np.max(ploty))
        leftx = left_fit.astype(float)
        rightx = right_fit.astype(float)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        

        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters

       
               
        return left_curverad, right_curverad        
    
    
    
    
    
        
    def warpToColor(self,image, warped,left_fitx, right_fitx, ploty):
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        
    
        
#        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result
        
    
        
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
     
        else:
            img_mask = source +'*.jpg'     # create a mask using glob
            
        images = self.collectCalibImages(img_mask)
        
        return images
    
    

import collections

class Line_Stats(Pipeline_helper):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = []
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        self.last_leftx = None
        self.last_rightx = None
        self.last_y = None
        
        self.Leftx = collections.deque(maxlen=3 * 20)
        self.Rightx = collections.deque(maxlen=3 * 20)
    def collectRadius(self,radius):
        """
        Collect all radii
        """
        self.radius_of_curvature.append(radius)
   
    def checkLaneWidth(self, leftx, rightx,lefty):
        """
        Sanity check on lane wdith using convoluted centroids
        Inputs:
        
        leftx: left centroid
        rightx: right centroids
        """
        # fid differenc eof lane points
        differ = np.array(rightx) - np.array(leftx)
        
        # basic sanity check
        s = differ < 0
        # if difference is negative , quit
        if(s.any()):
            return False
            
        #  maximum lane deivation of 100 pixels equals roughly 0.5 m        
        lane_dev = 100
        
        lane_width = 700        # lane width in pixels
        
#        lane_start = rightx[0] -leftx[0]        #lane width at start
#        lane_end = rightx[-1] -leftx[-1]        #lne width at end
##        print(lane_start)
#        print(lane_end)
        # avergae lane width from centroids
        avg_lane = np.mean(np.array(rightx) - np.array(leftx))
#        print("Lane width minus avg lane ",avg_lane ,np.abs(lane_width - avg_lane))
        # Check for avergae lane width
        if (np.abs(lane_width - avg_lane) < lane_dev ):

                
                self.detected = True
                self.last_leftx = (leftx)
                self.last_rightx = (rightx)
                self.last_y = lefty
                
                left,right = self.fitCoeff(leftx,lefty,rightx,lefty)
                self.Leftx.append(left)                
                self.Rightx.append(right)
                
                return True
        else:
            return False
        
        
        
        
        
        
        
        def checkRadius(self):
            return []
 
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    