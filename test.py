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

def finding_minimap(image, lines):
    #get the hight,lenth of the image.
    y, x, c = image.shape
    #initialize the boudary coordinates(outside of the image)
    ver_boudary = x
    mapcentre_x = x
    hor_boudary = y
    mapcentre_y = y
    map = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            if (abs(x1-x2)<3):#verdical boudary
                xmin=min(x1,x2)
                if (xmin<ver_boudary):
                    ver_boudary=xmin
                    mapcentre_x = int((xmin+x)/2)
            if (abs(y1-y2)<3):#horizontal boudary
                ymin=min(y1,y2)
                if (ymin<hor_boudary):
                    hor_boudary=ymin
                    mapcentre_y = int((ymin+y)/2)
    #display the boudaries on the map
    #horizontal
    cv2.line(map, (ver_boudary, hor_boudary), (x, hor_boudary), (0, 255, 0), 3)
    #verdical
    cv2.line(map, (ver_boudary, hor_boudary), (ver_boudary, y), (0, 255, 0), 3)
    
    #display the centre of the map
    centre=(mapcentre_x,mapcentre_y)
    print(centre)
    cv2.circle(map, centre, 5, (0,255,255), -2)
    return map

image = cv2.imread('./testImage/replay-tool.jpg')
canny = CannyEdge(image)#edge detection
cropped_Image = region_of_interest(canny)#get the mini map area

#set up hough transformation parameters
rho = 2
theta = np.pi/180
threshold = 80
#Hough Transformation
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=10, maxLineGap=5)
#Get the lines
map_info = finding_minimap(image, lines)
#Display the minmap edges/boudaries in the original pic
combo_image = cv2.addWeighted(map_info, 0.8, image, 1, 1)
#showing the image
'''
cv2.imshow('result',combo_image)
cv2.waitKey(0)
'''
plt.imshow(combo_image)
plt.show()