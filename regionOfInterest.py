import pyttsx3
import numpy as np
import cv2 #openCV

# Extract Region of Interest
def extract_roi(frame, fh, fw, right_side, left_side):
    rh = fh // 2
    rw = fw // 3
    if (right_side and left_side):
        x = (fw // 2) - (rw // 2)
    elif (right_side):
        x = 10
    else:
        x = fw - 10
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

def main():
    # Which side of the screen should ROI be on?
    # Set both to True if you want it in the center of the screen
    right_side = True
    left_side = False

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
            if (right_side and left_side):
                x = (fw // 2) - (rw // 2)
            elif (right_side):
                x = 10
            else:
                x = fw - 10
            y = (fh // 2) - (rh // 2)

            # Draw rectangle to show user where ROI is located on screen
            cv2.rectangle(frame, (x, y), (x+rw, y+rh), (255, 0, 0), 2)

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


if __name__ == "__main__":
    # execute only if run as a script
    main()