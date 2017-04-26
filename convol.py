# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:02:34 2017

@author: iiss
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# Read in a thresholded image
#warped = mpimg.imre1ad('warped_example.jpg')
#warped
# window settings
#window_width = 50 
#window_height = 80 # Break image into 9 vertical layers since image height is 720
#margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window_y = []
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(7*warped.shape[0]/8):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum)) - window_width/2
    r_sum = np.sum(warped[int(7*warped.shape[0]/8):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2+int(warped.shape[1]/2)
    
    l_centre_y = int((warped.shape[0] - int(3*warped.shape[0]/4))/2.0 ) + int(3*warped.shape[0]/4)
#    
        
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    window_y.append( l_centre_y)  
    
    return window_centroids,window_y
    
def find_all_centroids(warped, window_centroids, window_y, window_width, window_height, margin):   
    window = np.ones(window_width)
    
    l_center = window_centroids[0][0]
    r_center = window_centroids[0][1]

# Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        window_y.append(int(warped.shape[0]-(level+1)*window_height))
        # Find the best left centroid by using past left center as a reference
        	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        	    # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        
#        print(r_min_index)
#        print(r_max_index)
#        print()
        
        	    # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
        leftx = [item[0] for item in window_centroids]
        rightx = [item[1] for item in window_centroids]
        lefty = window_y
        righty = window_y
        

    
    return leftx, lefty,rightx,righty

#leftx, lefty,rightx,righty= find_window_centroids(warped, window_width, window_height, margin)
#
#
#left_fit = np.polyfit(lefty, leftx, 2)
#right_fit = np.polyfit(righty, rightx, 2)
### If we found any window centers
#if len(window_centroids) > 0:
#
#    # Points used to draw all the left and right windows
#    l_points = np.zeros_like(warped)
#    r_points = np.zeros_like(warped)
#
#    # Go through each level and draw the windows 	
#    for level in range(0,len(window_centroids)):
#        # Window_mask is a function to draw window areas
#	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
#	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
#	    # Add graphic points from window mask here to total pixels found 
#	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
#	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
#
#    # Draw the results
#    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
#    zero_channel = np.zeros_like(template) # create a zero color channel
#    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
#    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
#    output = cv2.addWeighted(warpage, 0.5, template, 0.2, 0.0) # overlay the orignal road image with window results
# 
## If no window centers found, just display orginal road image
#else:
#    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
#
## Display the final results
#plt.imshow(output)
#plt.title('window fitting results')
#plt.show()






#ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
#left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#
##out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
##out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#plt.imshow(warped)
#plt.plot(left_fitx, ploty, color='red')
#plt.plot(right_fitx, ploty, color='red')
#plt.xlim(0, 1280)
#plt.ylim(720, 0)

