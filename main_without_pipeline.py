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
from Pipeline_helper import Pipeline_helper
import convol as con



calib_image_dir = "camera_cal/"
test_image_dir = "test_images/"

NX = 9
NY = 6

window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

font = cv2.FONT_HERSHEY_SIMPLEX




# Make an instance of the pipeline
pipe = Pipeline_helper(NX,NY) 

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
thresh_col_s = [170,255]
#thresh_col_l = [200,255]


for im in range(len(output)):

    imge =  np.array(output[im])
    
    # Caluclate graients
    a,b,c,d = pipe.calcGradients(imge, absx, absy, mag, abs_dir )
    
       #perform filtering using color threshold
    s_binary = pipe.colorThr_s(imge,thresh_col_s)
#    s_binary = pipe.colorThr(imge,thresh_col_s,thresh_col_l)
#    
    combined_binary = pipe.getBinaryImg( a,b,c,d, s_binary )
    
    warped,img = pipe.getWarpedImg(imge, combined_binary)   
    
    leftx , lefty , rightx, righty= con.find_window_centroids(warped, window_width, window_height, margin)

    left_fitx, right_fitx, ploty = pipe.fitLine(warped,leftx, lefty,rightx,righty)
    
    
    l,r = pipe.getRadiusCurve(np.array(left_fitx), np.array(right_fitx),ploty)
    
    radius = float("{0:.2f}".format((l+r)/2.0))
     
    string = "Radius of Curvature is = "+str(radius)+" (m)"
    
    n_img = pipe.warpToColor(imge, warped,left_fitx, right_fitx, ploty)
    cv2.putText(n_img,string,(300,150), font, 1,(255,255,255),2)
   

    
    
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(combined_binary)
    

    
    ax1.set_title('Sobel X')
    ax2.imshow(warped)
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')



    
    
    
    
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













