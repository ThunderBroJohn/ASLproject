"""
PROTOTYPE 1
Kaylee Hertzog, John Miller, James Call, Bretton Steiner
"""

#imports
import pyttsx3
import numpy as np
import cv2 #openCV
import regionOfInterest
import imageProcesses

#initialize Text to speach
# engine = pyttsx3.init()
# engine.setProperty('rate', 165)#normal human speach is about 150 wpm

#OUTPUT functions
#This function will run the text to speach output
# def speak(talkToMe):
#     engine.say(talkToMe)
#     engine.runAndWait()

#This function will write the translated letters to the screen.
def draw_text(image, txt, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    color = (0, 0, 255)
    thickness = cv2.FILLED

    #txt_size = cv2.getTextSize(txt, font_face, scale, thickness)
    image = cv2.putText(image, txt, pos, font_face, scale, color, 1, cv2.LINE_AA)
    return image


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

            #If look for letter is false show output but
            # don't look for new letter until timer resets
            #add !!!!!!!!!!!!
            lookForLetter = False

            #look for ASL letter or symbol in frame
            #this is version 1 looking for stills not gestures
            letterString += translateSymbol(frame, lookForLetter)


            #show frame
            frame = draw_text(frame, letterString, (10,40))

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

