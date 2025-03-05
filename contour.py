import cv2
import numpy as np

def calculate_edge_lengths(approx):
    edge_lengths = []
    num_points = len(approx)

    for i in range(num_points):
        # Get current and next point (wrap around for the last edge)
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % num_points][0]

        # Calculate Euclidean distance between the two points
        edge_length = np.linalg.norm(np.array(pt1) - np.array(pt2))
        edge_lengths.append(edge_length)

    return edge_lengths

#Read the image and convert it to grayscale
image = cv2.imread('drone/untitled5.jpg')
image = cv2.GaussianBlur(image, (5, 5), 0)
image = cv2.resize(image, None, fx=0.5,fy=0.5)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Now convert the grayscale image to binary image
ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Now detect the contours
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

#Visualize the data structure
print("Length of contours {}".format(len(contours)))
print("Number of contours detected:",len(contours))
max_area= 0
Tempcounter= 0
pentIndex = 0

for cnt in contours:
    approx = list(cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True))
    (x,y)=cnt[0,0]
   

    if (len(approx) == 5): 
        edge_lengths = calculate_edge_lengths(approx)
        avg_length = np.mean(edge_lengths)
        area= cv2.contourArea(cnt)
        if(area >= 5000):
            print(f'pentagon area:{area} at index: {Tempcounter}')
            if(max_area <= area):
                max_area = area
            
                pentIndex = Tempcounter


    else:
        print('nope')
    
    Tempcounter += 1

# draw contours on the original image
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, contours, pentIndex, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

#Visualizing the results
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Drawn Contours', image_copy)
cv2.imshow('Binary Image', binary)
print(f'pentagon index is {pentIndex}')
# print(ret)
# print(pentagon)
cv2.waitKey(0)
cv2.destroyAllWindows()

