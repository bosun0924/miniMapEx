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
    hor_boudary = []
    ver_boudary = []
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            if (abs(x1-x2)<3):
                
            else if (abs(y1-y2)<3):
            #cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
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
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#showing the image
plt.imshow(line_image)
plt.show()