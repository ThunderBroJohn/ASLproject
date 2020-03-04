#ASL Translation project
#Kaylee Hertzog, John Miller, James Call, Bretton Steiner

#imports
import pyttsx3
import numpy as np
import cv2 #openCV

# Code taken from http://creat-tabu.blogspot.com/2013/08/opencv-python-hand-gesture-recognition.html

def main():
    # Creating camera object
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read the frame
        ret, img = cap.read()

        # Apply filters to clean up image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find the contours
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the largest contour
        max_area = 0
        for i in range(len(contours)):  
            cont = contours[i]
            area = cv2.contourArea(cont)
            if (area > max_area):
                max_area = area
                ci = i
        cont = contours[ci]

        # Draw the convex hull
        hull = cv2.convexHull(cont)

        # Calculate centr
        moments = cv2.moments(cont)
        if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00

        centr = (cx, cy)

        # Display the largest contour and convex hull
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [cont], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        # Find convexity defects
        hull = cv2.convexHull(cont, returnPoints=False)
        defects = cv2.convexityDefects(cont, hull)

        # Plot defects
        min_d = 0
        max_d = 0
        i = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(cont[s][0])
            end = tuple(cont[e][0])
            far = tuple(cont[f][0])
            dist = cv2.pointPolygonTest(cont, centr, True)
            cv2.line(drawing, start, end, [0, 255, 0], 2)
            cv2.circle(drawing, far, 5, [0, 0, 255], -1)
        # print(i)

        # Display the frame
        cv2.imshow('input', drawing)
        k = cv2.waitKey(10)
        if k == 27:
            break

if __name__ == "__main__":
    # execute only if run as a script
    main()
