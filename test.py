import cv2
import numpy as np
import matplotlib.pyplot as plt
def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage
def region_of_interest(image): 
    #get the resolution of the image
    height, width = image.shape
    #set the cropping polygons
    triangle = np.array([[(490, height),(490, 255),(width, 255),(width, height),]], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def display_lines(image, lines):
    #get the hight,lenth of the image.
    y, x, c = image.shape
    #initialize the boudary coordinates(outside of the image)
    ver_boudary = [[x,0],[x,y]]
    hor_boudary = [[0,y],[x,y]]
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            if (abs(x1-x2)<3):#verdical boudary
                xmin=min(x1,x2)
                if (xmin<ver_boudary[0][0]):
                    ver_boudary[0][0]=xmin
                    ver_boudary[1][0]=xmin
            if (abs(y1-y2)<3):#horizontal boudary
                ymin=min(y1,y2)
                if (ymin<hor_boudary[0][1]):
                    hor_boudary[0][1]=ymin
                    hor_boudary[1][1]=ymin
    #display the boudaries on the map
    #horizontal
    cv2.line(line_image, (hor_boudary[0][0], hor_boudary[0][1]), (hor_boudary[1][0], hor_boudary[1][1]), (0, 128, 255), 5)
    #verdical
    cv2.line(line_image, (ver_boudary[0][0], ver_boudary[0][1]), (ver_boudary[1][0], ver_boudary[1][1]), (0, 255, 128), 5)
    return line_image

lane_image = cv2.imread('./testImage/replay-tool.jpg')
canny = CannyEdge(lane_image)#edge detection
cropped_Image = region_of_interest(canny)#get the mini map area

#set up hough transformation parameters
rho = 2
theta = np.pi/180
threshold = 80
#Hough Transformation
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=10, maxLineGap=5)
#Get the lines
line_image = display_lines(lane_image, lines)
#Display the minmap edges/boudaries in the original pic
combo_image = cv2.addWeighted(line_image, 0.8, lane_image, 1, 1)
#showing the image
cv2.imshow('result',combo_image)
cv2.waitKey(0)