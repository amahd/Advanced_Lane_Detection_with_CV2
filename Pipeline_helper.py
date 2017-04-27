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


calib_image_dir = "camera_cal/"
test_image_dir = "test_images/"

# Coordinates for source and detination points

s_x = [580,700,1040,260]

s_y = [460,460,680,680]
    
d_x = [260,1040,1040,260]

d_y = [0,0,720,720]

src = np.int32([ [s_x[0],s_y[0] ], [ s_x[1],s_y[1]], 
                                         [s_x[2],s_y[2]], 
                                         [s_x[3],s_y[3]]])
        

dst = np.int32([ [d_x[0],d_y[0] ], [ d_x[1],d_y[1]], 
                                         [d_x[2],d_y[2]], 
                                         [d_x[3],d_y[3]]])

class Pipeline_helper():
       
    def __init__(self, a,b):
        """ Initializes the pipeline """
        self.NX = a     # Number of corners on x-axis for calibration
        self.NY = b     # # Number of corners on y-axis for calibration
        self.grid = (a,b)
        print("Class Initilaized")
        
    def cameraCalibMatrices(self,img):
        """
        takes a list of images as input and finds true chessboard corners
        Inputs:
        Img: Input test image list
        Returns:
        mtx,dtx : outputs matrices for calibration
        
        """
        
        print("Getting to know the corners of images")
        imgp = []
        objp = []
        positives = 0
        objponints = np.zeros((self.NX * self.NY,3),np.float32)
        
        objponints[:,:2] = np.mgrid[ 0:self.NX , 0:self.NY].T.reshape(-1,2)
        
        for ind, row in enumerate(img):
        
            #First convert image to grascale
            gray = cv2.cvtColor( row, cv2.COLOR_RGB2GRAY )
        
            # getting chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
            
            if (ret == True):  # Only take those images who are correctly detected
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
         Inputs:
         Img: Input image to undistort
         mtx,dst. matrices required to undistrort   
         val: True to supress output plot
         Returns: Undistorted image
         """
        
         udst = cv2.undistort(img, mtx, dst, None, mtx)
         if (val):
             return udst
        
         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
         f.tight_layout()
         ax1.imshow(img)
         ax1.set_title('Original Image', fontsize=10)
         ax2.imshow(udst)
         ax2.set_title('Undistorted Image', fontsize=10)
         plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        
         return udst
        
   
    def absSobelThresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        
        """"
        Apply absolute threshold using x or y derivtaive
        
        Inputs
        image: Input image
        orient: Orientation for applying derivative on right axis
        sobel_kernel: Sobel kernel value (odd number)
        thresh: a list with minimum and max value of thresholds
        
        Returns: binary image 
        """
        gray = image
        # Check orientation
        if orient == 'x':
            	abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
       
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
       
        # Create a copy and apply the threshold
        abs_binary = np.zeros_like(scaled_sobel)
      
        # Create a binary image of ones where threshold is met, zeros otherwise
        abs_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[-1])] = 1
        return abs_binary
    
    def magThresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Apply magnitude threshold using x and y derivtaive
        
        Inputs
        image: Input image
        
        sobel_kernel: Sobel kernel value (odd number)
        thresh: a list with minimum and max value of thresholds
        
        Returns: binary image 
        """
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
    
    def dirThreshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Apply magnitude threshold using x and y derivtaive
        
        Inputs
        image: Input image
        
        sobel_kernel: Sobel kernel value (odd number)
        thresh: a list with minimum and max value of thresholds
            
        Returns: binary image 
        """
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

    
    
    def calcGradients(self, image, absx,absy, mag, abs_dir):
        """
        A wrapper function to perform all gradient based threshold detections
        
        Inputs
        iamge: Input images
        absx: Sobel kernel and thresholds for detection on xaxis
        absy: Sobel kernel and thresholds for detection on yaxis
        mag:  Sobel kernel and thresholds for magnitude threshold dectection
        abs_dir: Sobel kernel and thresholds for directional threshold detection
        
        Returns
        Binary images form each edge detection method
        """
        
#        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # using The R Channel as it is more robust for color detection
        gray = image[:,:,0]
        
        # Clean up the image by using Gaussian blur
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        # Absolute edge with gradient on xaxis
        abs_x =  self.absSobelThresh( blur, 'x', absx[0], absx[1])
        
        # Absolute edge with gradient on yaxis
        abs_y =  self.absSobelThresh( blur, 'y', absy[0], absy[1])
        
        #Magnitude based edge detection
        mag_x =  self.magThresh(blur, mag[0], mag[1])
        
        # Directional threshold call
        dir_x =  self.dirThreshold(blur, abs_dir[0], abs_dir[1])
         
        return abs_x, abs_y, mag_x, dir_x
         
    
    
    
    def colorThr_s(self,image,s_thresh):
        """
        Edge detection on color channel
        Inputs
        image: input image
        s_thresh:  color threshold value
        
        Retruns
        a binary image of ones where threshold is met, zeros otherwise
        """
        
        
        # Convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Get the s channel as its most robust
        s_channel = hls[:,:,2]

        # Apply threshold 
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        return s_binary
    
    
    
    
    def getBinaryImg (self, xd, yd, mg ,di ,s_binary ):
        """
        Get final binary image with using edge and color filtering
        
        Inputs
        xd: Binary images after applying x-axis gradient 
        yd: Binary images after applying y-axis gradient 
        mg: Binary images after applying magnitude gradient 
        di: Binary images after applying directional gradient 
        s_binary:  binay image with color threshold  detection
        
        Retruns
        a binary image with all edge and color filtering
        """
        final = np.zeros_like(xd)
        # Apply derivates filtering
        final[((xd == 1) & (yd == 1)) | ((mg == 1) & (di == 1))] = 1
    
        # Apply color filter for s_channel
        binary = np.zeros_like(final)
        binary[(s_binary == 1) | (final == 1)] = 1
        
        
        return binary
    
    def getIntermImg (self, xd, yd, mg ,di ):
        """
        Get intermediate binary image with combined usage of edge detection filtering
        
        Inputs
        xd: Binary images after applying x-axis gradient 
        yd: Binary images after applying y-axis gradient 
        mg: Binary images after applying magnitude gradient 
        di: Binary images after applying directional gradient 
       
        
        Retruns
        a binary image with all edge  filtering
        """
        final = np.zeros_like(xd)
        # Apply derivates filtering
        final[((xd == 1) & (yd == 1)) | ((mg == 1) & (di == 1))] = 1
    
 
        return final
    
    
    

           
    def getWarpedImg(self,imge, combined_binary):
        
        """
        Get the warped image
        Inputs
        imge: Input image
        combined_binary: A binary image with all color and gradient threshold filtering
        
        Returns
        warped: Warped image
        img: Original unwared image
        """
        
        img_size = (imge.shape[1], imge.shape[0])
          
        img = np.copy(imge)
        
        #Plot points on the images to warp
        cv2.polylines(img,np.int32([src]),True,(255,0,0), 5)       
        
        #Perform prespective transform using src and dst points defined at the top of page
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective( combined_binary, M, img_size)
        

        return warped, img
    

    def fitVector(self, warped ,left_fit,right_fit):
        """
        Calculates the right and left lane lines coefficients using polyfit
        Inputs
        waped: input warped image
        left_fit: coefficients for left lane line fitting
        right_fit: coefficients for left lane line fitting
        
        Returns
        left_fitx: Left lane line
        right_fitx: right lane line
        ploty: refernce line for lane drawing
        """
    
        # refernce line
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        
        #Polynominal fitting using coeffs from input
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        return left_fitx, right_fitx, ploty
    
    def getRadiusCurve(self,left_fit, right_fit, ploty):
        
        """
        Calculates the radus of the curve made by the lane lines
        Inputs
        left_fitx: Left lane line
        right_fitx: right lane line
        ploty: refernce line for lane drawing
        
        Returns
        radii of the two lane lines
        """
        
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
               
        return left_curverad, right_curverad        
    
    def warpToColor(self,image, warped,left_fitx, right_fitx, ploty):
        """
        Calculates the radus of the curve made by the lane lines
        Inputs
        Image: original image
        warped image: warped binary image
        left_fitx: Left lane line
        right_fitx: right lane line
        ploty: refernce line for lane drawing
        
        Returns
        unwarped image with a color block on it to show lane lines
        """        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Get inverse perspetive tansform perspective image
        Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        
#        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result
    
    
    def getCameraOffset(self, img,left_fit,right_fit):
        """
        Get how much off the car is form the centre of the lane
        Inputs
        Img: Image othe car on the road
        left_fit: left lane line fit
        right_fit: right lane line fit
            
        Returns
        offset in meters
        
        """
        camera_position = img.shape[1]/2    # ideal camera position in image centre
        lane_centre = (left_fit[-1] + right_fit[-1])/2 # projected image centre
        centre_offset_pixels = abs(camera_position - lane_centre) # offste in pixels
        centre_offset_m = float("{0:.2f}".format((centre_offset_pixels)* 3.7/700))  #offste in meters  
    
        return centre_offset_m
        
    def collectRefImages(self,mask):
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
        # check which images dir to read
        if (source == calib_image_dir):
            img_mask = source +'calibration*.jpg'     # create a mask using glob
     
        else:
            img_mask = source +'*.jpg'     # create a mask using glob
            
        images = self.collectRefImages(img_mask)    # reading images from dir
        
        return images
    
    

import collections

class Line_Sanity():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # Save the last line coefficients if they are sane 
        self.last_leftx = None
        self.last_rightx = None
        self.last_y = None
        
        self.Leftx = collections.deque(maxlen=3 * 10)
        self.Rightx = collections.deque(maxlen=3 * 10)
        
    def collectRadius(self,radius):
        """
        Collect all radii
        """
        self.radius_of_curvature.append(radius)
   
    def checkLaneWidth(self, leftx, lefty,rightx,righty):
        """
        Sanity check on lane wdith using convoluted centroids
        Inputs:
        
        leftx: left centroid
        rightx: right centroids
        lefty: y coordinate fro right lane
        righty: y coordinate for right lane
        
        """
        # fid differenc of lane points
        differ = np.array(rightx) - np.array(leftx)
        
        # basic sanity check, differnece shold not be negative
        s = differ < 0
        # if difference is negative , quit
        if(s.any()):
            return False
            

        # Individual values of x coordinatesshould also nt be negative
        s = np.array(leftx) < 0
        if(s.any()):
            return False
        
        # Individual values of x coordinates should not cross onto other half of figure 
        # ie it must not exceed 640 pixels in this case 
        s = np.array(leftx) >=  640   
        if(s.any()):
            return False       
        
        s = np.array(rightx) <=  640 
        if(s.any()):
            return False       
        
        lane_dev = 100 # max lane deviation in pixels
        
        lane_width = 700        # lane width in pixels
        
        # avergae lane width from centroids
        avg_lane = np.mean(np.array(rightx) - np.array(leftx))

        # Check for avergae lane width
        if (np.abs(lane_width - avg_lane) < lane_dev ):

                
            
                self.detected = True
                
                # save original points as last detcted values
                self.last_leftx = (leftx)
                self.last_rightx = (rightx)
                self.last_y = lefty
                
                # Calc and save coeffs in the ring buffer
                left,right = fitCoeff(leftx,lefty,rightx,righty)
                self.Leftx.append(left)                
                self.Rightx.append(right)
                
                return True
        else:
            return False
        
        
        
        
        
        
       
    
        
        
def fitCoeff(leftx, lefty,rightx ,righty):
        """
        Calculates the right and left lane lines coefficients using polyfit
        Inputs
        leftx: x coordinates for left lane points
        lefty: y coordinates for left lane points
        rightx: x coordinates for right lane points
        righty:   y coordinates for right lane points
        
        Returns_ coefficients for line fitting
        """
        
        # Obtaining Coeffs for left and right lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    