# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:02:34 2017


"""

import numpy as np



def find_window_centroids(warped, window_width, window_height):
    
    """
    Do convolution to find initial lane  vertices for right and lift
    
    Inputs:
    warped : warped image
    window_width: Width of window to convolve
    window_height: height of convolution window
    
    Returns:
    window_centroids: a list of x coordinates of left and right lane    
    l_centre_y. y coordinates of the above x coorindates
    """
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window_y = []
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # 1/8 bottom of image to get slice, 
    l_sum = np.sum(warped[int(7*warped.shape[0]/8):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum)) - window_width/2
    r_sum = np.sum(warped[int(7*warped.shape[0]/8):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2+int(warped.shape[1]/2)
    
    # y-coordinate for the above xcoordinates is taken to be the centre of last 1/8 of the image
    l_centre_y = int((warped.shape[0] - int(7*warped.shape[0]/8))/2.0 ) + int(7*warped.shape[0]/8)
    
        
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    window_y.append( l_centre_y)  
    
    return window_centroids,window_y
    
def find_all_centroids(warped, window_centroids, window_y, window_width, window_height, margin):   
   
    """ 
    Use Initial coordinates to search the entire image for x coordinates for right and left lane lines 
    
    Inputs:
    warped : warped image
    window_width: Width of window to convolve
    window_height: height of convolution window
    window_centroids: a list of x coordinates of left and right lane 
    margin: Area to look into rather than look around the whole line

    Returns:
    leftx: x coordinates for left lane points
    lefty: y coordinates for left lane points
    rightx: x coordinates for right lane points
    righty:   y coordinates for right lane points
    """
    
    
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
        
        # add to what we found for that layer
        window_centroids.append((l_center,r_center))
        
        leftx = [item[0] for item in window_centroids]
        rightx = [item[1] for item in window_centroids]
        lefty = window_y
        righty = window_y

    return leftx, lefty,rightx,righty
