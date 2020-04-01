
#imports
#import pyttsx3 #USE VERSION 2.71
import numpy as np
from matplotlib import pyplot as plt
import cv2 #openCV
#import regionOfInterest
#import resize
import imageProcesses

def initialize_comparison_library():
    #abc... and bs(backspace) and space
    alphabetList = [(cv2.imread("ASLproject/ASLproject/edgePreprocess/a1.png", 0), "a")]
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/b1.png", 0),"b"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c1.png", 0),"c"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c2.png", 0),"c"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/d1.png", 0),"d"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/e1.png", 0),"e"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/f1.png", 0),"f"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/g1.png", 0),"g"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/h1.png", 0),"h"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/i1.png", 0),"i"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/j1.png", 0),"j"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/k1.png", 0),"k"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/l1.png", 0),"l"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/m1.png", 0),"m"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/n1.png", 0),"n"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o1.png", 0),"o"]))
    #alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o2.png", 0),"o"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/p1.png", 0),"p"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/q1.png", 0),"q"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/r1.png", 0),"r"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/s1.png", 0),"s"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/t1.png", 0),"t"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/u1.png", 0),"u"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/v1.png", 0),"v"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/w1.png", 0),"w"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0),"x"]))
    #alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x2.png", 0),"x"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/y1.png", 0),"y"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/z1.png", 0),"z"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/_1.png", 0),"_"]))#space or _
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/bs1.png", 0),"bs"]))
    #test = "alphabetList loaded with " + str(len(alphabetList)) + " items"
    #print(test)
    #print(alphabetList[0][1])#a
    return alphabetList

def findMatchesBetweenImages(image_1, image_2, nf=500, sf=1.2, wta=2, st=cv2.ORB_HARRIS_SCORE, ps=31):
    matches = None       # type: list of cv2.DMath
    image_1_kp = None    # type: list of cv2.KeyPoint items
    image_1_desc = None  # type: numpy.ndarray of numpy.uint8 values.
    image_2_kp = None    # type: list of cv2.KeyPoint items.
    image_2_desc = None  # type: numpy.ndarray of numpy.uint8 values.

    orb = cv2.ORB_create(nfeatures=nf, scaleFactor=sf, WTA_K=wta, scoreType=st, patchSize=ps)

    # START ******************************************************
    # solve for rotation, scale, lighting

    # orb reference https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb
    #initiate ORB detector
    #orb = cv2.ORB()

    #to gray
    #image_1_gray = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    #image_2_gray = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

    #could have blur

    #find keypoints
    image_1_kp = orb.detect(image_1,None)
    image_2_kp = orb.detect(image_2,None)

    #compute the descriptors
    image_1_kp, image_1_desc = orb.compute(image_1, image_1_kp)
    image_2_kp, image_2_desc = orb.compute(image_2, image_2_kp)

    #compute matches
    #referenced https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(image_1_desc,image_2_desc)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    #img3 = cv2.drawMatches(image_1,image_1_kp,image_2,image_2_kp,matches[:10], flags=2)
    #plt.imshow(img3),plt.show()


    # END ********************************************************

    # I coded the return statement for you. You are free to modify it -- just
    # make sure the tests pass.
    return image_1_kp, image_2_kp, matches#[:15]



def compare_images(image1,image2,distanceThreshold=0.85):
    #cv2.ORB
    orb = cv2.ORB_create()

    kp1 = orb.detect(image1,None)
    kp2 = orb.detect(image2,None)

    kp1, des1 = orb.compute(image1, kp1)
    kp2, des2 = orb.compute(image2, kp2)

    #testing
    #img1 = cv2.drawKeypoints(image1, kp1, image1, color=(0,255,0), flags=0)
    #img2 = cv2.drawKeypoints(image2, kp2, image2, color=(0,255,0), flags=0)
    #cv2.imshow("test",img1)
    #cv2.waitKey(0)
    #cv2.imshow("test",img2)
    #cv2.waitKey(0)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1,des2, k=2)
    #kp1, kp2, matches = findMatchesBetweenImages(image1,image2)

    # Apply ratio test
    good = []
    bad = []
    for m,n in matches:
        #print("M, N distance ", m.distance, n.distance)
        if(m.distance < 100):
            if(m.distance < distanceThreshold*n.distance):#.75*n
                good.append([m])
            else:   
                bad.append([m])
        else:
            bad.append([m])


    #testing
    print(len(good))
    print(len(bad))
    percent_match = len(good)/(len(good)+len(bad))
    #print(percent_match)
    
    return percent_match


def find_match(alphabetList,image):
    mirrorImage = cv2.flip(image, 1)

    bestMatch = 0
    temp1 = 0
    temp2 = 0
    letterGuess = ""
    for testLetter in alphabetList:
        temp1 = compare_images(testLetter[0],image)
        temp2 = compare_images(testLetter[0],mirrorImage)
        if(temp1 > temp2):
            if(bestMatch < temp1):
                bestMatch = temp1
                letterGuess = testLetter[1]
        else:
            if(bestMatch < temp2):
                bestMatch = temp2
                letterGuess = testLetter[1]
        test = str(bestMatch) + ", " + letterGuess
        print(test)
    if(bestMatch > 0.10):
        return letterGuess
    else:
        return ""
    
alphabetList = initialize_comparison_library()

#image1 = cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0)
#image2 = cv2.imread("ASLproject/ASLproject/edgePreprocess/x2.png", 0)

testImage = cv2.imread("ASLproject/ASLproject/edgePreprocess/o2.png", 0)

testResults = find_match(alphabetList,testImage)
print(testResults)

# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = image1
#img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,image1,flags=2)
#cv2.imshow("testout",img3)
#img3 = cv2.drawMatches(image_1,image_1_kp,image_2,image_2_kp,matches[:10], flags=2)
    #plt.imshow(img3),plt.show()

#print(kp1, kp2)

#centeroids1, corners1 = imageProcesses.compute_harris_corner(image)
#centeroids2, corners2 = imageProcesses.compute_harris_corner(image2)



#print(str(int(centeroids[0][0])) + ", " + str(int(centeroids[0][1]))) 

#res1 = np.hstack((centeroids1,corners1))
#res1 = np.int0(res)
#print(res1) 
#image[res[:,1],res[:,0]]=[0,0,255] #RED
#image[res[:,3],res[:,2]] = [0,255,0] #GREEN

#cv2.imshow("test", image)
