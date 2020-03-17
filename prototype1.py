"""
PROTOTYPE 1
Kaylee Hertzog, John Miller, James Call, Bretton Steiner
"""

#imports
import pyttsx3
import numpy as np
import cv2 #openCV
import regionOfInterest
import resize
import imageProcesses

#initialize Text to speach
# engine = pyttsx3.init()
# engine.setProperty('rate', 165)#normal human speach is about 150 wpm

#OUTPUT functions
#This function will run the text to speach output
# def speak(talkToMe):
#     engine.say(talkToMe)
#     engine.runAndWait()




"""
This function will take in an image and compare it to
a small file based data set of ASL symbols, and return 
what letter it is.

IF lookForLetter is false, we will wait for it to be true
before looking for the next letter.

returns letter (and lookForLetter to false if found? (Look reset))
"""
def translateSymbol(frame, lookForLetter):
    if(not lookForLetter):
        return ""
    #ADD TRANSLATION LOGIC HERE
    
def image_compare(image1, image2):
    sum_diff = np.sum(cv2.absdiff(image1, image2))
    return sum_diff

def main():

    #string for use in output
    letterString = "test"
    lookForLetter = True

    #capture computer camera
    cap = cv2.VideoCapture(0)

    #run translation program
    while(True):
        ret, frame = cap.read()

        if (ret):
            #first flip image <--> people work better with mirrors
            frame = cv2.flip(frame, 1)

            #Get Region of Interest
            frame, roi = regionOfInterest.extract_roi(frame)
            if (roi is not None):
                roi = imageProcesses.sobel_gradient_edge(roi)
                roi = resize.normalize_image_size(roi)#500 by 500

            #If look for letter is false show output but
            # don't look for new letter until timer resets
            #add !!!!!!!!!!!!
            lookForLetter = False

            #look for ASL letter or symbol in frame
            #this is version 1 looking for stills not gestures
            letterString += translateSymbol(frame, lookForLetter)


            #show frame
            frame = imageProcesses.draw_text(frame, letterString, (10,40))

            cv2.imshow("Prototype 1", frame)
            key = cv2.waitKey(1)

        #At end of loop check for keyboard input
        if key and key == ord('r'): #recalibrate
            regionOfInterest.calibrate(frame)
            key = None
        if key and key == ord('c'): #clear
            letterString = "" #reset
            key = None
        if key and key == ord('s'): #speak
            # speak(letterString)
            letterString = "" #reset
            key = None
        if key and key == ord('q'): #quit
            key = None
            break

    cv2.destroyAllWindows()


#run main
if __name__ == "__main__":
    main()

