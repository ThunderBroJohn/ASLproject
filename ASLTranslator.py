"""
ASL Translator
Kaylee Hartzog, John Miller, James Call, Bretton Steiner
"""

# Imports
import pyttsx3 #USE VERSION 2.71
import numpy as np
import cv2 # openCV
import regionOfInterest
import resize
import imageProcesses
import imageCompare

# Initialize Text to Speech
engine = pyttsx3.init()
engine.setProperty('rate', 165)

def most_likely_match(matchList):
    letter = ""

    if (matchList is not None and matchList != []):
        letterList = []
        # First, try to find a letter that matched more than the others
        for tempLetter, _ in matchList:
            letterList.append(tempLetter)
        labels, counts = np.unique(letterList, return_counts=True)
        highestLetter = ""
        highestCount = 0
        for index in np.arange(len(labels)):
            if (counts[index] > 1 and (counts[index] > highestCount or highestLetter == "")):
                highestLetter = labels[index]
                highestCount = counts[index]

        # If that didn't work, then go by highest match percentage
        if highestLetter == "":
            highestPercentage = 0.0
            for tempLetter, tempPercentage in matchList:
                if tempPercentage > highestPercentage:
                    highestLetter = tempLetter
                    highestPercentage = tempPercentage

        # Return the most likely letter
        letter = highestLetter

    return letter

def speak(talkToMe):
    engine.say(talkToMe)
    engine.runAndWait()

def main():
    # Initialize variables
    letterString = "hello world"
    letter = ""
    delayTimerDuration = 20
    delayTimer = delayTimerDuration
    matchList = []

    # Capture camera
    cap = cv2.VideoCapture(0)

    # Run translation program
    while(True):
        ret, frame = cap.read()

        if (ret):
            # First flip image - people work better with mirrors
            frame = cv2.flip(frame, 1)

            # Get Region of Interest
            frame, roi = regionOfInterest.extract_roi(frame)

            if (roi is not None):
                # Convert roi into threshold of hand shape
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                ret, roi = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

                # Resize roi to 500 by 500 pixels
                roi = resize.normalize_image_size(roi)
                cv2.imshow("ROI", roi)

                # Get letter and match percentage
                tempLetter, tempPercent = imageCompare.compareToLibrary(roi)

                # Append results to list
                if (tempLetter is not None and tempPercent > 0.0):
                    matchList.append((tempLetter, tempPercent))

                # Debug log
                if (tempLetter is not None and tempPercent > 0.0):
                    print(f"{tempLetter} : {tempPercent:0.2f} %")
                else:
                    print("No match")

            # Draw text onto frame
            frame = imageProcesses.draw_text(frame, letterString, (10, 40))
            frameWidth, _, _ = frame.shape
            frame = imageProcesses.draw_text(frame, letter, (frameWidth - 10, 40))

            # Show frame
            cv2.imshow("Prototype 1", frame)
            
            # Check for keyboard input
            key = cv2.waitKey(1)

        # Count the delayTimer
        if delayTimer > 0:
            delayTimer -= 1
        else:
            letter = most_likely_match(matchList)
            delayTimer = delayTimerDuration

        # Switch roi location
        if key and key == ord('l'):
            regionOfInterest.switch_square_location()
            key = None
        # Approve letter
        if key and key == ord('a') and letter != "":
            # If backspace symbol detected remove item from string
            if(letter == "bs"):
                if(len(letterString) != 0): 
                    letterString = letterString[:-1]
            else:
                letterString = letterString + letter
            matchList = []
            delayTimer = delayTimerDuration
            letter = ""
            key = None
        # Clear
        if key and key == ord('c'):
            letterString = "" # Reset
            key = None
        # Speak
        if key and key == ord('s'):
            if(letterString != ""):
                speak(letterString)
            letterString = "" # Reset
            key = None
        # Quit
        if key and key == ord('q'):
            key = None
            break

    cv2.destroyAllWindows()


#run main
if __name__ == "__main__":
    main()
