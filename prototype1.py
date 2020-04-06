"""
PROTOTYPE 1
Kaylee Hartzog, John Miller, James Call, Bretton Steiner
"""

#imports
import pyttsx3 #USE VERSION 2.71
import numpy as np
import cv2 #openCV
import regionOfInterest
import resize
import imageProcesses
#import imageCompare

#initialize Text to speach
engine = pyttsx3.init()
engine.setProperty('rate', 165)#normal human speach is about 150 wpm


#OUTPUT functions
#This function will run the text to speach output
def speak(talkToMe):
    engine.say(talkToMe)
    engine.runAndWait()




"""
This function will take in an image and compare it to
a small file based data set of ASL symbols, and return 
what letter it is.
IF lookForLetter is false, we will wait for it to be true
before looking for the next letter.
returns letter (and lookForLetter to false if found? (Look reset))
"""
    
def image_compare(image1, image2):
    sum_diff = np.sum(cv2.absdiff(image1, image2))
    return sum_diff

def translateSymbol(frame, lookForLetter, alphabetList):
    if(not lookForLetter):
        return ""
    #difftest = image_compare(frame,alphabetList[0][0])
    #print(difftest)
    guess = imageProcesses.find_match(alphabetList, frame)
    return guess
            


def main():
    alphabetListSobel = imageProcesses.initialize_comparison_library_sobel()
    alphabetListBandW = imageProcesses.initialize_comparison_library_BandW()

    #string for use in output
    letterString = "test"
    lookForLetter = True
    saveRoi = False
    thresholdValue = 160

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
                # roi = imageProcesses.sobel_gradient_edge(roi)
                roi2 = roi
                roi2 = imageProcesses.sobel_gradient_edge(roi2,(7,7))
                roi2 = resize.normalize_image_size(roi2)
                #cv2.imshow("ROI Sobel",roi2)

                #testLetter = imageProcesses.find_match(alphabetListSobel,roi2)
                #print(testLetter)

                # Testing to get threshold
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                #invert = cv2.bitwise_not(blur)
                ret, roi = cv2.threshold(blur, thresholdValue, 255, cv2.THRESH_BINARY)


                roi = resize.normalize_image_size(roi)#500 by 500
                cv2.imshow("ROI", roi)


                testGuess = imageProcesses.group_sorted_comparison(roi,roi2,alphabetListSobel,alphabetListBandW)
                print(testGuess)

                #letter, percent = imageCompare.compareToLibrary(roi)

                #if (letter is not None and percent > 70.0):
                #    print(f"{letter} : {percent:0.2f} %")
                #elif (letter is not None and percent > 0.0):
                #    print(f"Maybe {letter} : {percent:0.2f} %")
                #else:
                #    print("No match")

                #if (saveRoi):
                #    cv2.imwrite("roi.png", roi)
                #    saveRoi = False

            #If look for letter is false show output but
            # don't look for new letter until timer resets
            #add !!!!!!!!!!!!
            lookForLetter = False

            #look for ASL letter or symbol in frame
            #this is version 1 looking for stills not gestures
            if(roi is not None):
                tempLetter = translateSymbol(roi, lookForLetter, alphabetListBandW)
            else:
                tempLetter = ""
            if(tempLetter == "bs"): #If backspace symbol detected remove item from string
                if(len(letterString) != 0): 
                    letterString = letterString[:-1]
            else:
                letterString = letterString + tempLetter


            #show frame
            frame = imageProcesses.draw_text(frame, letterString, (10,40))

            cv2.imshow("Prototype 1", frame)
            
            key = cv2.waitKey(1)

        #At end of loop check for keyboard input
        if key and key == ord('r'): #recalibrate
            regionOfInterest.calibrate(frame)
            key = None
        if key and key == ord('t'): #recalibrate
            regionOfInterest.toggle_tracking()
            key = None
        if key and key == ord('l'): #recalibrate
            regionOfInterest.switch_square_location()
            key = None
        if key and key == ord('c'): #clear
            letterString = "" #reset
            key = None
        if key and key == ord('s'): #speak
            speak(letterString)
            letterString = "" #reset
            key = None
        if key and key == ord('-'): #speak
            if (thresholdValue > 5):
                thresholdValue -= 5
            key = None
        if key and key == ord('='): #speak
            if (thresholdValue < 250):
                thresholdValue += 5
            key = None
        if key and key == ord('p'): #speak
            saveRoi = True
            key = None
        if key and key == ord('q'): #quit
            key = None
            break

    cv2.destroyAllWindows()


#run main
if __name__ == "__main__":
    main()