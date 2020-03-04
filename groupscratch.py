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

def run_camera_test():
    """ Live capture your laptop camera """
    cap = cv2.VideoCapture(0)  # Notice the '0' instead of a filename
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
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
    #run_camera_test()

    letterString = ""

    #init()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    while(True):
        #detect camera input
        

        """ Comparison portion
            This is where we will 
            1 detect the hand from the camera
            2 compare the hand against the alphabet photo library to find a match
            3 add letter to letterString
            4 be able to put spaces between letters (thread timer, bool?)
        """

        if cv2.waitKey(1) & 0xFF == ord('r'):
            pass #recalebrate
        if cv2.waitKey(1) & 0xFF == ord('c'):
            letterString = ""
        if cv2.waitKey(1) & 0xFF == ord('s'):
            speak(letterString)
            letterString = ""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    # execute only if run as a script
    main()
