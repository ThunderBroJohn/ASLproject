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
        frame, _, _ = automatic_brightness_and_contrast(frame,1)
        #hsv_frame = cv2.colorChange(frame,cv2.COLOR_BGR2HSV)#for hand histogram
        #gray_frame = cv2.colorChange(frame,cv2.COLOR_BGR2GRAY)#for comparison

        if (ret):
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
    #tts_test()
    run_camera_test()

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
