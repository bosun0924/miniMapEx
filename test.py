import cv2
import numpy as np
import matplotlib.pyplot as plt
def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage
def region_of_interest(image): 
    height, width = image.shape
    triangle = np.array([[(490, height),(490, 255),(width, 255),(width, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return line_image

lane_image = cv2.imread('./testImage/replay-tool.jpg')
canny = CannyEdge(lane_image)
cropped_Image = region_of_interest(canny)
rho = 2
theta = np.pi/180
threshold = 80
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=10, maxLineGap=5)
line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
plt.imshow(line_image)
plt.show()