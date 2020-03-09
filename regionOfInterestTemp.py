import pyttsx3
import numpy as np
import cv2 #openCV

# Extract Region of Interest
def extract_roi(frame, fh, fw, right_side, left_side):
    rh = fh // 2
    rw = fw // 3
    if (right_side and not left_side):
        x = 10
    elif (not right_side and left_side):
        x = fw - 10 - rw
    else:
        x = (fw // 2) - (rw // 2)
    y = (fh // 2) - (rh // 2)
    roi = np.zeros((rh, rw, 3), dtype=np.uint8)
    roi = frame[y:y+rh, x:x+rw]

    return roi

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
        ret, thresh1 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
        return drawing

def main():
    # Which side of the screen should ROI be on?
    # Set both to True or both to False if you want it in the center of the screen
    right_side = False
    left_side = True

    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while(cap.isOpened()):
        #detect camera input
        ret, frame = cap.read()

        if ret == True:
            # Extract the region of interest
            roi = extract_roi(frame, fh, fw, right_side, left_side)
            rh, rw, _ = roi.shape
            if (right_side and not left_side):
                x = 10
            elif (not right_side and left_side):
                x = fw - 10 - rw
            else:
                x = (fw // 2) - (rw // 2)
            y = (fh // 2) - (rh // 2)

            # Draw rectangle to show user where ROI is located on screen
            cv2.rectangle(frame, (x, y), (x+rw, y+rh), (255, 0, 0), 2)

            # Use contour detection on roi
            roi = contourDetection(roi)

            # Flip images for user convenience
            roi = cv2.flip(roi, 1)
            frame = cv2.flip(frame, 1)

            # Create resulting frame
            frame = draw_images(roi, frame)

            # Display the resulting frame
            cv2.imshow("ROI Feedback", frame)
            cv2.waitKey(1000 // fps)
        else:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    # execute only if run as a script
    main()