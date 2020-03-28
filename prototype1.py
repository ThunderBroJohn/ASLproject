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

#initialize Text to speach
engine = pyttsx3.init()
engine.setProperty('rate', 165)#normal human speach is about 150 wpm


#This function pulls preproccessed images for use in comparison
def initialize_comparison_library():
    #abc... and bs(backspace) and space
    alphabetList = [(cv2.imread("ASLproject/edgePreprocess/a1.png", 0), "a")]
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/b1.png", 0),"b"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/c1.png", 0),"c"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/c2.png", 0),"c"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/d1.png", 0),"d"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/e1.png", 0),"e"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/f1.png", 0),"f"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/g1.png", 0),"g"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/h1.png", 0),"h"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/i1.png", 0),"i"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/j1.png", 0),"j"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/k1.png", 0),"k"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/l1.png", 0),"l"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/m1.png", 0),"m"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/n1.png", 0),"n"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/o1.png", 0),"o"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/o2.png", 0),"o"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/p1.png", 0),"p"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/q1.png", 0),"q"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/r1.png", 0),"r"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/s1.png", 0),"s"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/t1.png", 0),"t"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/u1.png", 0),"u"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/v1.png", 0),"v"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/w1.png", 0),"w"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/x1.png", 0),"x"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/x1.png", 0),"x"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/y1.png", 0),"y"]))
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/z1.png", 0),"z"]))

    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/_1.png", 0)," "]))#space or _
    alphabetList.append(([cv2.imread("ASLproject/edgePreprocess/bs1.png", 0),"bs"]))
    #test = "alphabetList loaded with " + str(len(alphabetList)) + " items"
    #print(test)
    #print(alphabetList[0][1])#a
    return alphabetList


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
    difftest = image_compare(frame,alphabetList[0][0])
    print(difftest)
    """
    #broken please fix
    #ADD TRANSLATION LOGIC HERE
    #minMatch = number 
    temp = [0, ""]
    for letter in alphabetList:
        diff = image_compare(frame,letter[0])
        diff2 = image_compare(frame,cv2.flip(letter[0],1))
        if(diff2 < diff):
            diff = diff2 #take the better comparison of left or right

        if(diff < temp[0]):
            temp = [diff, letter[1]]
    #if(temp[0] > minMatch):#??? working on this logic
    #   return ""
    return temp[1]   
    """
            


def main():
    alphabetList = initialize_comparison_library()

    print(alphabetList[0][1])
    
    # pic = cv2.imread("ASLproject/edgePreprocess/a1.png", 0)
    # cv2.imshow("p", pic)
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
                cv2.imshow("ROI",roi)
                # print("I made it")

            #If look for letter is false show output but
            # don't look for new letter until timer resets
            #add !!!!!!!!!!!!
            lookForLetter = False

            #look for ASL letter or symbol in frame
            #this is version 1 looking for stills not gestures
            if(roi is not None):
                tempLetter = translateSymbol(roi, lookForLetter, alphabetList)
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

            closestNum = None
            closestLetter = None

            #TODO change Frame from color to Soble
            # yeah, I know this is stupidly simple to the point that it might not be a good idea/ or work
            # but the idea is that you take the frame, (resize it and the item in alphabetList) and you do
            # the image_compare with each item in the alphabetList. whichever has the least difference, it
            # will print that letter

            for item in alphabetList:
                pic = item[0]
                pic = resize.normalize_image_size(pic)
                frame = resize.normalize_image_size(frame)
                # pic = cv2.resize(pic, (500, 500), interpolation=cv2.INTER_AREA)
                # frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)

                diff = image_compare(pic, frame)
                print(diff)
            #     if closestNum == None:
            #         closestNum = diff
            #         closestLetter = item[1]
            #     if diff < closestNum:
            #         closestNum = diff
            #         closestLetter = item[1]
            
            # print("LETTER: ", closestLet)



        #At end of loop check for keyboard input
        if key and key == ord('r'): #recalibrate
            regionOfInterest.calibrate(frame)
            key = None
        if key and key == ord('c'): #clear
            letterString = "" #reset
            key = None
        if key and key == ord('s'): #speak
            speak(letterString)
            letterString = "" #reset
            key = None
        if key and key == ord('q'): #quit
            key = None
            break

    cv2.destroyAllWindows()


#run main
if __name__ == "__main__":
    main()

