#ASL Translation project
#Kaylee Hertzog, John Miller, James Call, Bretton Steiner

#imports
import pyttsx3
import numpy as np
import cv2 #openCV

#initialize Text to speach
engine = pyttsx3.init()
engine.setProperty('rate', 165)#normal human speach is about 150 wpm

def tts_test():
#text to speach usage
    engine.say("Hello, how are you today?")
    engine.runAndWait()

#folowing functions from https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def gaussian_kernel(size=5, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
    #alternatively just use src = cv.GaussianBlur(src, (5, 5), 0)
    #gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

def sobel_gradient_edge(gray):
    #Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)#kernal X direction
    #Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)#kernal Y direction
    #Ix = ndimage.filters.convolve(img, Kx) translate from scipy image processing
    #Iy = ndimage.filters.convolve(img, Ky)
    #G = np.hypot(Ix, Iy)
    #G = G / G.max() * 255
    #theta = np.arctan2(Iy, Ix)
    #return (G, theta)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

    

#funciton from https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

        # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def run_camera_test():
    """ Live capture your laptop camera """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #frame, _, _ = automatic_brightness_and_contrast(frame,1)
        #hsv_frame = cv2.colorChange(frame,cv2.COLOR_BGR2HSV)#for hand histogram
        #frame = cv2.colorChange(frame,cv2.COLOR_BGR2GRAY)#for comparison gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#breaks... why?
        frame = sobel_gradient_edge(frame)

        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def speak(talkToMe):
    engine.say(talkToMe)
    engine.runAndWait()


def main():
    tts_test()
    #run_camera_test()

    letterString = ""

    #init()

    #cap = cv2.VideoCapture(0)
    #ret, frame = cap.read()

    #while(True):
        #detect camera input
        

    #    """ Comparison portion
    #        This is where we will 
    #        1 detect the hand from the camera
    #        2 compare the hand against the alphabet photo library to find a match
    #        3 add letter to letterString
    #        4 be able to put spaces between letters (thread timer, bool?)
    #    """

    #    if cv2.waitKey(1) & 0xFF == ord('r'):
    #        pass #recalebrate
    #    if cv2.waitKey(1) & 0xFF == ord('c'):
    #        letterString = ""
    #    if cv2.waitKey(1) & 0xFF == ord('s'):
    #        speak(letterString)
    #        letterString = ""
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break




if __name__ == "__main__":
    # execute only if run as a script
    main()
