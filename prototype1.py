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


#Function IN PROGRESS
def initialize_comparison_library():
    #abc... and bs(backspace) and space
    alphabetList = [(cv2.imread("ASLproject/ASLproject/edgePreprocess/a1.png"), "a")]
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/b1.png"),"b"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c1.png"),"c"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c2.png"),"c"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/d1.png"),"d"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/e1.png"),"e"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/f1.png"),"f"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/g1.png"),"g"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/h1.png"),"h"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/i1.png"),"i"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/j1.png"),"j"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/k1.png"),"k"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/l1.png"),"l"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/m1.png"),"m"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/n1.png"),"n"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o1.png"),"o"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o2.png"),"o"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/p1.png"),"p"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/q1.png"),"q"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/r1.png"),"r"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/s1.png"),"s"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/t1.png"),"t"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/u1.png"),"u"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/v1.png"),"v"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/w1.png"),"w"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png"),"x"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png"),"x"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/y1.png"),"y"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/z1.png"),"z"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/_1.png"),"_"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/bs1.png"),"bs"]))
    return alphabetList


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
    
def image_compare(image1, image2):
    sum_diff = np.sum(cv2.absdiff(image1, image2))
    return sum_diff

def translateSymbol(frame, lookForLetter, alphabetList):
    if(not lookForLetter):
        return ""
    #ADD TRANSLATION LOGIC HERE
        minMatch = num 
        for letter in alphabetList:
            pass #WRITE COMPARE
            


def main():
    alphabetList = initialize_comparison_library()

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

