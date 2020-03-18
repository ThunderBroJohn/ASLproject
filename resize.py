import math
import numpy as np
import cv2 #openCV

SIDE_LENGTH = 500

def fix_to_ratio(image):
    # Get dimensions of image
    width = image.shape[1]
    height = image.shape[0]

    # Take the larger side length and make image into square shape
    # by adding additional black space
    if height > width:
        background = np.zeros((height, height, 3), dtype=np.uint8)
        gap = math.floor((height - width) / 2)
        background[:, gap:-gap] = image
    elif height < width:
        background = np.zeros((width, width, 3), dtype=np.uint8)
        gap = math.floor((width - height) / 2)
        background[gap:-gap, :] = image
    else:
        background = image

    return background

def normalize_image_size(image):
    # Add black space to image until ratio is met
    image = fix_to_ratio(image)

    # Resize image
    resized = cv2.resize(image, (SIDE_LENGTH, SIDE_LENGTH), interpolation=cv2.INTER_AREA)

    # Return resized image
    return resized

def main():
    # For testing purposes
    image = cv2.imread("reference.png")
    image = normalize_image_size(image)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#run main
if __name__ == "__main__":
    main()