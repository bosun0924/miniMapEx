import cv2
import numpy as np
import matplotlib.pyplot as plt
def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage
def region_of_interest(image): 
    '''
    height = image.shape[0]
    triangle = np.array([[(200, height),(550, 250),(1100, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    '''
    height, width = image.shape
    triangle = np.array([[(490, height),(490, 255),(width, 255),(width, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

lane_image = cv2.imread('./testImage/replay-tool.jpg')
canny = CannyEdge(lane_image)
cropped_Image = region_of_interest(canny)
print(canny.shape)
plt.imshow(cropped_Image)
plt.show()