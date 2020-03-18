#ASL Translation project
#Kaylee Hertzog, John Miller, James Call, Bretton Steiner

#imports
import numpy as np
import cv2 #openCV

hand_hist = None
is_hand_hist_created = False
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=5)

    rows, cols = thresh.shape
    top = None
    bottom = None
    left = None
    right = None
    sumcols = np.sum(thresh, axis=0)
    sumrows = np.sum(thresh, axis=1)

    i = 0
    while (top is None):
        if (i >= len(sumrows)):
            break
        elif (sumrows[i] != 0):
            top = i
        i += 1

    i = rows - 1
    while (bottom is None):
        if (i <= 0):
            break
        elif (sumrows[i] != 0):
            bottom = i
        i -= 1

    i = 0
    while (left is None):
        if (i >= len(sumcols)):
            break
        elif (sumcols[i] != 0):
            left = i
        i += 1

    i = cols - 1
    while (right is None):
        if (i <= 0):
            break
        elif (sumcols[i] != 0):
            right = i
        i -= 1

    # For debugging:
    # print(f"Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}")

    # This portion will crop until only the Region of Interest remains
    # mask = np.zeros((rows, cols), dtype=np.uint8)
    # mask[top : bottom, left : right] = np.full((bottom - top, right - left), 255, dtype=np.uint8)
    # mask = cv2.merge((mask, mask, mask))
    # final = cv2.bitwise_and(frame, mask)

    # This portion will keep only the Region of Interest
    roi = None
    if (top is not None and bottom is not None and right is not None and left is not None):
        roi = np.zeros((bottom - top, right - left), dtype=np.uint8)
        roi = frame[top : bottom, left : right]

        # This portion will keep the whole frame, but draw a rectangle representing the Region of Interest
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    return frame, roi

def calibrate(frame):
    global is_hand_hist_created, hand_hist
    if is_hand_hist_created:
        is_hand_hist_created = False
    else:
        is_hand_hist_created = True
        hand_hist = hand_histogram(frame)

def extract_roi(frame):
    global is_hand_hist_created, hand_hist
    if is_hand_hist_created:
        frame, roi = hist_masking(frame, hand_hist)
    else:
        frame = draw_rect(frame)
        roi = None

    return frame, roi

def main():
    global hand_hist
    global is_hand_hist_created
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            frame = hist_masking(frame, hand_hist)

        else:
            frame = draw_rect(frame)

        # frame = cv2.flip(frame, 1)
        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
        

if __name__ == "__main__":
    # execute only if run as a script
    main()
