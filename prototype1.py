"""
PROTOTYPE 1
Kaylee Hertzog, John Miller, James Call, Bretton Steiner
"""

#imports
import pyttsx3
import numpy as np
import cv2 #openCV

#initialize Text to speach
engine = pyttsx3.init()
engine.setProperty('rate', 165)#normal human speach is about 150 wpm

"""
recalibrate function
This function will calebrate or recalebrate
to lighing and skintone for hand recognition
"""
def recalibrate():
    #!!!!!!!!!!!!!!!!
    pass


#OUTPUT functions
#This function will run the text to speach output
def speak(talkToMe):
    engine.say(talkToMe)
    engine.runAndWait()

#This function will write the translated letters to the screen.
def drawTextToScreen(frame, showText):
    #add in draw text to screen!!!!!!!!!!!!! Kaylee's work
    return frame


"""
This function will take in an image and compare it to
a small file based data set of ASL symbols, and return 
what letter it is.

IF lookForLetter is false, we will wait for it to be true
before looking for the next letter.

returns letter (and lookForLetter to false if found? (Look reset))
"""
def translateSymbol(frame, lookForLetter)
    if(!lookForLetter):
        return ""
    #ADD TRANSLATION LOGIC HERE
    


def main():

    #string for use in output
    letterString = ""
    lookForLetter = True

    #capture computer camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    #calebrate!!!!!

    #run translation program
    while(ret == True):
        #first flip image <--> people work better with mirrors
        frame = cv2.flip(frame, 1)

        #If look for letter is false show output but 
        # don't look for new letter until timer resets
        #add !!!!!!!!!!!!

        #look for ASL letter or symbol in frame
        #this is version 1 looking for stills not gestures
        letterString += translateSymbol(frame, lookForLetter)


        #show frame
        frame = drawTextToScreen(frame, letterString)

        #At end of loop check for keyboard input
        if cv2.waitKey(1) & 0xFF == ord('r'): #recalebrate
            pass #recalebrate goes here
        if cv2.waitKey(1) & 0xFF == ord('c'): #clear
            letterString = "" #reset
        if cv2.waitKey(1) & 0xFF == ord('s'): #speak
            speak(letterString) 
            letterString = "" #reset
        if cv2.waitKey(1) & 0xFF == ord('q'): #quit
            break


#run main
if __name__ == "__main__":
    main()

