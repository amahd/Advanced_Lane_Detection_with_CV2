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
from Pipeline_helper import Pipeline_helper,Line_Stats
#from Line_Stats import Line_Stats

import convol as con

from moviepy.editor import VideoFileClip
from IPython.display import HTML

calib_image_dir = "camera_cal/"
test_image_dir = "test_images/"

NX = 9
NY = 6

window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

font = cv2.FONT_HERSHEY_SIMPLEX

VALX = 40
VALY = 40
VALM = 50
s = 3.0

absx = [5, [VALX,VALX*s]]
absy = [5, [VALY,VALY*s]]
mag = [5, [VALM,VALM*s]]
abs_dir = [25, [0.9,1.1]]
thresh_col_s = [100,255]
thresh_col_l = [220,255]

line = Line_Stats()
indx = 0
#def pipeline(image):
#    
#    
#    imge =  np.array(image)
#    
#    # Caluclate graients
#    a,b,c,d = pipe.calcGradients(imge, absx, absy, mag, abs_dir )
#    
#    #perform filtering using color threshold
#    s_binary = pipe.colorThr(imge,thresh_col)
#    
#    combined_binary = pipe.getBinaryImg( a,b,c,d, s_binary )
#    
#    warped,img = pipe.getWarpedImg(imge, combined_binary)    
#    
#    leftx , lefty , rightx, righty= con.find_window_centroids(warped, window_width, window_height, margin)
#
#    left_fitx, right_fitx, ploty = pipe.fitLine(warped,leftx, lefty,rightx,righty)
#    
#    
#    l,r = pipe.getRadiusCurve(np.array(left_fitx), np.array(right_fitx),ploty)
#    
#    radius = float("{0:.2f}".format((l+r)/2.0))
#    line_stat.collectRadius(radius) 
#    string = "Radius of Curvature is = "+str(radius)+" (m)"
#    
#    n_img = pipe.warpToColor(imge, warped,left_fitx, right_fitx, ploty)
#    cv2.putText(n_img,string,(300,150), font, 1,(255,255,255),2)
#    
##    return n_img, warped, left_fitx, right_fitx, ploty,img
#    return n_img

def pipeline(image):
    centre_x = []
    centre_y = []
    imge =  np.array(image)
    global indx
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
    
       
    val = line.checkLaneWidth(leftx,rightx,righty)    
        
#    elif (line.last_leftx is not None):
#            left_p, right_p = pipe.fitCoeff(warped,line.last_leftx, line.last_lefty,rightx,righty)
#            left_fitx, right_fitx, ploty = pipe.fitVector(warped,left_p,right_p)



    if (len(line.Leftx) == line.Leftx.maxlen):
#        left_p, right_p = pipe.fitCoeff(warped,leftx, lefty,rightx,righty)
        cum_left = np.array(line.Leftx)
        cum_right = np.array(line.Rightx)
        left_p = np.median(cum_left,axis = 0)
        right_p = np.median(cum_right,axis = 0)
        
    else:
        left_p, right_p = pipe.fitCoeff(leftx, lefty,rightx,righty)
        
    left_fitx, right_fitx, ploty = pipe.fitVector(warped,left_p,right_p)
    l,r = pipe.getRadiusCurve(np.array(left_fitx), np.array(right_fitx),ploty)
    
    radius = float("{0:.2f}".format((l+r)/2.0))
#    print(radius) 

#    print()
    string = "Radius of Curvature is = "+str(radius)+" (m)"
    
    n_img = pipe.warpToColor(imge, warped,left_fitx, right_fitx, ploty)
    if (indx//10):
        cv2.putText(n_img,string,(300,150), font, 1,(255,255,255),2)
    indx += 1

    return n_img


# Make an instance of the pipeline
pipe = Pipeline_helper(NX,NY) 
line_stat = Line_Stats()
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




#for im in range(len(output)):
#
#    n_img,warped, left_fitx,right_fitx,ploty,img = pipeline(output[im])
#   
#    f, ((ax1, ax2,ax3)) = plt.subplots(1, 3, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(img)
#    
#    ax1.set_title('Sobel X')
#    ax2.imshow(warped)
#    ax2.plot(left_fitx, ploty, color='red')
#    ax2.plot(right_fitx, ploty, color='red')
#
#
#    ax3.imshow(n_img)
#
#    break
#    


white_output = 'output.mp4'

clip2 = VideoFileClip('project_video.mp4')#.subclip(0,85)
yellow_clip = clip2.fl_image(pipeline)
yellow_clip.write_videofile(white_output, audio=False)










    
    
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













