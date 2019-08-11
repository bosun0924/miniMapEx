import cv2
import numpy as np
import matplotlib.pyplot as plt
def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #gray = image[...,2]
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 100)
    return cannyImage

def region_of_interest(image): 
    #get the resolution of the image
    height, width = image.shape
    #set up the map extracting area
    map_height_limit = int(0.7*height)
    map_width_limit = int(0.82*width)
    #set the cropping polygons
    crop_area = np.array([[(map_width_limit, height),(map_width_limit, map_height_limit),(width, map_height_limit),(width, height),]], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, 255)
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

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image
'''
#image = cv2.imread('./testImage/4.jpg')
image = cv2.imread('./testImage/5.png')
image = cv2.resize(image,(800,450))
canny = CannyEdge(image)#edge detection
cropped_Image = region_of_interest(canny)#get the mini map area

#set up hough transformation parameters
rho = 2
theta = np.pi/180
threshold = 80
#Hough Transformation
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=20, maxLineGap=1)
#Get the lines
map_info = finding_minimap(image, lines)
hough = display_lines(image, lines)
#edge_hough = cv2.addWeighted(canny, 0.8, hough, 1, 1)
#hough_mapinfo=cv2.addWeighted(map_info, 0.8, hough, 1, 1)
#Display the minmap edges/boudaries in the original pic
combo_image = cv2.addWeighted(map_info, 0.8, image, 1, 1)
#showing the image
canny_cvt = cv2.cvtColor(canny,cv2.COLOR_GRAY2RGB)
ana = cv2.addWeighted(canny_cvt, 0.8, hough, 1, 1)

cv2.imshow('result',combo_image)
cv2.waitKey(0)

plt.figure()
plt.imshow(image[...,1])

plt.figure()
plt.imshow(cropped_Image)

plt.figure()
plt.imshow(hough)

plt.figure()
plt.imshow(ana)

plt.figure()
plt.imshow(combo_image)
plt.show()
'''
cap = cv2.VideoCapture("./testImage/test.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame,(800,450))
    canny = CannyEdge(frame)
    cropped_Image = region_of_interest(canny)
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([ ]), minLineLength=40, maxLineGap=5)
    #line_image = display_lines(frame, lines)
    map_info = finding_minimap(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, map_info, 1, 1)
    cv2.imshow("Image", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#'''