# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:29:16 2017

@author: Aneeq Mahmood
email: aneeq.sdc@gmail.com
"""

# Importing regular modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.close('all')
#importing pipeline
from Pipeline_helper import Pipeline_helper, Line_Stats
import convol as con

def hsvfil(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_yellow  = np.array([ 0, 80, 200])
    upper_yellow = np.array([ 40, 255, 255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    return res





calib_image_dir = "camera_cal/"
test_image_dir = "test/"

NX = 9
NY = 6

window_width = 100 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 150 # How much to slide left and right for searching

font = cv2.FONT_HERSHEY_SIMPLEX




# Make an instance of the pipeline
pipe = Pipeline_helper(NX,NY)
line = Line_Stats()

""" Collect all images for camera calibration"""

images = pipe.collectImages(calib_image_dir)
test_images = pipe.collectImages(test_image_dir)

#import sys
#sys.exit()
import os
if os.path.exists("Coeff.npy"):
    print("Using existing Coefficients ")
    coeff = np.load('Coeff.npy') 
    mat = np.load('Mat.npy')
else:
    print("Calculaitng new distortion Coefficients ")
    mat,coeff = pipe.cameraCalibMatrices(images)
    np.save('Coeff.npy',coeff)
    np.save('Mat.npy',mat)


output = []
#test_image = images[10]
for j in range(len(test_images)):
    output.append(pipe.cameraDistremove(test_images[j],mat,coeff,True))

VALX = 40
VALY = 40
VALM = 50
s = 3.0

absx = [5, [VALX,VALX*s]]
absy = [5, [VALY,VALY*s]]
mag = [5, [VALM,VALM*s]]
abs_dir = [25, [0.9,1.1]]
thresh_col_s = [150,255]
thresh_col_l = [220,255]

indx = 0

for im in range(len(output)):
    centre_x = []
    centre_y = []
    imge =  np.array(output[im])
    
    # Caluclate graients
    a,b,c,d = pipe.calcGradients(imge, absx, absy, mag, abs_dir )
    
       #perform filtering using color threshold
    s_binary = pipe.colorThr_s(imge,thresh_col_s)
#    s_binary = pipe.colorThr(imge,thresh_col_s,thresh_col_l)
#    
    combined_binary = pipe.getBinaryImg( a,b,c,d, s_binary )
    
    warped,img = pipe.getWarpedImg(imge, combined_binary)   

    centre_x, centre_y = con.find_window_centroids(warped, window_width, window_height, margin)
    
    leftx , lefty , rightx, righty = con.find_all_centroids(warped, centre_x, centre_y,
                                                            window_width, window_height, margin)
    
#    if (line.checkLaneWidth(leftx,rightx,righty)):   # Check on lane width
    
 
    val = line.checkLaneWidth(leftx, lefty,rightx,righty)    
        
    if (len(line.Leftx) == line.Leftx.maxlen):
#        left_p, right_p = pipe.fitCoeff(warped,leftx, lefty,rightx,righty)
        cum_left = np.array(line.Leftx)
        cum_right = np.array(line.Rightx)
        left_p = np.mean(cum_left,axis = 0)
        right_p = np.mean(cum_right,axis = 0)
       
    elif (val):
        left_p, right_p = pipe.fitCoeff(leftx, lefty,rightx,righty)
    elif (line.last_leftx is not None):
        left_p, right_p = pipe.fitCoeff(line.last_leftx, line.last_y,line.last_rightx, line.last_y)
    else:
        continue
        
        
        
    left_fitx, right_fitx, ploty = pipe.fitVector(warped,left_p,right_p)
    l,r = pipe.getRadiusCurve(np.array(left_fitx), np.array(right_fitx),ploty)
    
    camera_position = imge.shape[1]/2
    print(camera_position)
    lane_center = (left_fitx[719] + right_fitx[719])/2
    print(lane_center)
    center_offset_pixels = abs(camera_position - lane_center)* 3.7/700
    print(center_offset_pixels )
        
    radius = float("{0:.2f}".format((l+r)/2.0))
    print(radius,val) 

    print()
    string1 = "Radius of Curvature is = "+str(radius)+" (m)"
    string2 = "Camera off centre by = "+str(center_offset_pixel)+" (m)"
    n_img = pipe.warpToColor(imge, warped,left_fitx, right_fitx, ploty)
    if (indx//10):
        cv2.putText(n_img,string,(300,150), font, 1,(255,255,255),2)
        cv2.putText(n_img,string,(300,150), font, 1,(255,255,255),2)
    indx += 1
    
    
    
    
    
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4))
    f.tight_layout()
    ax1.imshow(combined_binary,cmap='gray')
    

    
    ax1.set_title('Sobel X')
    ax2.imshow(warped,cmap='gray')
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')
    plt.plot(rightx,righty,'.')
    plt.plot(leftx,righty,'.')


    

    
    
    
    
#    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(a)
#    ax1.set_title('Sobel X')
#    ax2.imshow(b)
#    ax2.set_title('Sobel Y')
#    ax3.set_title('Mag threshold')
#    ax4.set_title('Direction Threshold')
#    ax3.imshow(c)
#    ax4.imshow(d)
#    plt.savefig('result_images/binary'+str(im)+'.jpg')
#
#    
#    
#    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(imge)
#    ax1.set_title('original')
#    ax2.imshow(final)
#    ax2.set_title('Binary Thresholds')
#    ax3.set_title('Color Threshold')
#    ax3.imshow(s_binary)
#    ax4.set_title('Final')
#    ax4.imshow(combined_binary)
#    plt.savefig('result_images/Final'+str(im)+'.jpg')    
#    













