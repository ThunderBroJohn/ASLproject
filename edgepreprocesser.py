import pyttsx3
import numpy as np
import glob
import os
import cv2 #openCV

def sobel_gradient_edge(gray):
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    #presets
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    #make sobel X and sobel Y
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    #combine sobel X and sobel Y
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

#############
def load_image_files(subpath):
    #path = os.getcwd() + '/' + subpath
    #files = [f for f in glob.glob(path + "**/*.jpg", recursive=False)]
    #ASLproject/ASLproject/
    files = [f for f in glob.glob("ASLproject/ASLproject/alphabet cropped/*.jpg", recursive=False)]
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

def preprocess_all_files():
    subpath = "alphabet cropped"
    files = load_image_files(subpath)

    print(files)
    num_files = len(files)
    print('Number of files =', num_files)

    if len(files) > 0:
        for i in range(0, len(files)):
            frame = cv2.imread(files[i], 0)#in as gray

            if frame is not None:
                # Use contour detection on roi
                roi = sobel_gradient_edge(frame)

                # # Create resulting frame
                # frame = draw_images(roi, frame)

                # Display the resulting frame
                # cv2.imshow("Result", draw_images(roi, frame))
                # cv2.imshow("Result", roi)
                # cv2.waitKey(100)

                # Write result file
                cv2.imwrite(f"ASLproject/ASLproject/edgePreprocess/{i}.png", roi)
            else:
                print("Frame is None.")

        cv2.destroyAllWindows()
    else:
        print("No files found.")

def main():

    # gray = cv2.imread("alphabet cropped/a1.jpg", 0)
    # blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # # equ = cv2.equalizeHist(blur)
    # invert = cv2.bitwise_not(blur)
    # ret, threshold = cv2.threshold(invert, 160, 255, cv2.THRESH_BINARY_INV)
    # # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # # ret, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow("Result", draw_images(threshold, blur))
    # cv2.waitKey(0)

    preprocess_all_files()

if __name__ == "__main__":
    # execute only if run as a script
    main()


