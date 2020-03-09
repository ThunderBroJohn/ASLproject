import pyttsx3
import numpy as np
import glob
import os
import cv2 #openCV

def load_image_files(subpath):
    path = os.getcwd() + '/' + subpath
    files = [f for f in glob.glob(path + "**/*.png", recursive=False)]
    files.sort()
    return files

def draw_images(image_1, image_2):
    """ Draws image_1 and image_2 next to each other.
    Args:
    image_1 (numpy.ndarray): The first image (can be color or grayscale).
    image_2 (numpy.ndarray): The image to search in (can be color or grayscale)

    Returns:
    output (numpy.ndarray): An output image that draws lines from the input
                            image to the output image based on where the
                            matching features are.
    """
    # Compute number of channels.
    num_channels = 1
    if len(image_1.shape) == 3:
        num_channels = image_1.shape[2]
    # Separation between images.
    margin = 10
    # Create an array that will fit both images (with a margin of 10 to
    # separate the two images)
    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                            image_1.shape[1] + image_2.shape[1] + margin,
                            3), dtype=np.uint8)
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image_1.shape[0],
                         :image_1.shape[1],
                         channel_idx] = image_1
            joined_image[:image_2.shape[0],
                         image_1.shape[1] + margin:,
                         channel_idx] = image_2
    else:
        joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
        joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    return joined_image

# Code taken from http://creat-tabu.blogspot.com/2013/08/opencv-python-hand-gesture-recognition.html
def contourDetection(img):
    # Apply filters to clean up image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        equ = cv2.equalizeHist(blur)
        invert = cv2.bitwise_not(equ)
        ret, thresh1 = cv2.threshold(invert, 160, 255, cv2.THRESH_BINARY)
        # ret, thresh1 = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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

        # # Find convexity defects
        # hull = cv2.convexHull(cont, returnPoints=False)
        # defects = cv2.convexityDefects(cont, hull)

        # # Plot defects
        # min_d = 0
        # max_d = 0
        # i = 0
        # for i in range(defects.shape[0]):
        #     s, e, f, d = defects[i,0]
        #     start = tuple(cont[s][0])
        #     end = tuple(cont[e][0])
        #     far = tuple(cont[f][0])
        #     dist = cv2.pointPolygonTest(cont, centr, True)
        #     cv2.line(drawing, start, end, [0, 255, 0], 2)
        #     cv2.circle(drawing, far, 5, [0, 0, 255], -1)
        # # print(i)

        rows, cols, _ = img.shape
        for row in range(0, rows):
            for col in range(0, cols):
                value = thresh1[row][col]
                drawing[row][col] = (value, value, value)

        return drawing

def preprocess_all_files():
    subpath = "alphabet cropped"
    files = load_image_files(subpath)

    print(files)
    num_files = len(files)
    print('Number of files =', num_files)

    if len(files) > 0:
        for i in range(0, len(files)):
            frame = cv2.imread(files[i])

            if frame is not None:
                # Use contour detection on roi
                roi = contourDetection(frame)

                # # Create resulting frame
                # frame = draw_images(roi, frame)

                # Display the resulting frame
                cv2.imshow("Result", draw_images(roi, frame))
                cv2.waitKey(100)

                # Write result file
                cv2.imwrite(f"preprocessed alphabet/{i}.png", roi)
            else:
                print("Frame is None.")

        cv2.destroyAllWindows()
    else:
        print("No files found.")

def main():

    # gray = cv2.imread("alphabet cropped/v2.png", 0)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # equ = cv2.equalizeHist(blur)
    # invert = cv2.bitwise_not(equ)
    # ret, threshold = cv2.threshold(invert, 160, 255, cv2.THRESH_BINARY)
    # # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # # ret, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow("Result", draw_images(threshold, gray))
    # cv2.waitKey(0)

    preprocess_all_files()

if __name__ == "__main__":
    # execute only if run as a script
    main()