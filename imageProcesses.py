#ASL Translation project
#Kaylee Hertzog, John Miller, James Call, Bretton Steiner
#imports
import numpy as np
import cv2 #openCV

#This function will write the translated letters to the screen.
def draw_text(image, txt, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    color = (0, 0, 255)
    #thickness = cv2.FILLED

    #txt_size = cv2.getTextSize(txt, font_face, scale, thickness)
    image = cv2.putText(image, txt, pos, font_face, scale, color, 1, cv2.LINE_AA)
    return image

#this function will create an image of blurred edges
def sobel_gradient_edge(image, blur=(5,5)):
    #blur image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#occasional hickups
    gray = cv2.GaussianBlur(gray, blur, 0)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Gradient-X
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    # Combine X and Y
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

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

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def compute_harris_corner(grayimage, threashold=10**(-4)): #GRAY IMAGES
    #1computer gradient and matrix R to store corner strengths?
    gray = np.float32(grayimage)
    dst = cv2.cornerHarris(gray,2,3,0,0.4)
    dst = cv2.dilate(dst,None)
    #2 threshold gets rid of weak features
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    #3 non-maximum suppresion to compute local maximum of features and discard others
    #find centeroids
    ret, lables, stats, centeroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centeroids),(5,5),(-1,-1),criteria)

    return centeroids, corners
    # Now draw them
    #res = np.hstack((centroids,corners))
    #res = np.int0(res)
    #img[res[:,1],res[:,0]]=[0,0,255] #RED
    #img[res[:,3],res[:,2]] = [0,255,0] #GREEN
    
    #4 compute discriptors for remaning with get_descriptor

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
    #print(len(good))
    #print(len(bad))
    percent_match = len(good)/(len(good)+len(bad))
    #print(percent_match)
    
    return percent_match

def find_match(alphabetList,image,distanceThreshold=0.85):
    mirrorImage = cv2.flip(image, 1)

    bestMatch = 0
    temp1 = 0
    temp2 = 0
    letterGuess = ""
    for testLetter in alphabetList:
        temp1 = compare_images(testLetter[0],image,distanceThreshold)
        temp2 = compare_images(testLetter[0],mirrorImage,distanceThreshold)
        if(temp1 > temp2):
            if(bestMatch < temp1):
                bestMatch = temp1
                letterGuess = testLetter[1]
        else:
            if(bestMatch < temp2):
                bestMatch = temp2
                letterGuess = testLetter[1]
        #test = str(bestMatch) + ", " + letterGuess
        #print(test)
    if(bestMatch > 0.10):
        return letterGuess
    else:
        return ""

#This function pulls preproccessed images for use in comparison
def initialize_comparison_library_sobel():
    #abc... and bs(backspace) and space
    #group 1 fist type a s t n m 
    alphabetList = [(cv2.imread("ASLproject/ASLproject/edgePreprocess/a1.png", 0), "a")]      #0
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/m1.png", 0),"m"])) #1
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/n1.png", 0),"n"])) #2
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/s1.png", 0),"s"])) #3
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/t1.png", 0),"t"])) #4

    #group 2 side finger up shape d i r u
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/d1.png", 0),"d"])) #5
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/i1.png", 0),"i"])) #6
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/r1.png", 0),"r"])) #7
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/u1.png", 0),"u"])) #8

    #group 3 circle type c o
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c1.png", 0),"c"])) #9
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/c2.png", 0),"c"])) #10
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o1.png", 0),"o"])) #11
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/o2.png", 0),"o"])) #12

    #group 4 pointing down types p q
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/p1.png", 0),"p"])) #13
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/q1.png", 0),"q"])) #14

    #group 5 many fingers up
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/b1.png", 0),"b"])) #15
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/f1.png", 0),"f"])) #16

    #group 6 sidewase point g h
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/g1.png", 0),"g"])) #17
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/h1.png", 0),"h"])) #18

    #group 7 split fingers up v k
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/v1.png", 0),"v"])) #19
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/k1.png", 0),"k"])) #20

    #group 8 unique shapes
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/e1.png", 0),"e"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/j1.png", 0),"j"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/l1.png", 0),"l"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/w1.png", 0),"w"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0),"x"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/x2.png", 0),"x"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/y1.png", 0),"y"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/z1.png", 0),"z"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/_1.png", 0)," "]))#space or _
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/edgePreprocess/bs1.png", 0),"bs"]))
    #test = "alphabetList loaded with " + str(len(alphabetList)) + " items"
    #print(test)
    #print(alphabetList[0][1])#a
    return alphabetList

#This function pulls preproccessed images for use in comparison
def initialize_comparison_library_BandW():
    #abc... and bs(backspace) and space
    #group 1 fist type a s t n m 
    alphabetList = [(cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/a1.png", 0), "a")]      #0
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/m1.png", 0),"m"])) #1
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/n1.png", 0),"n"])) #2
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/s1.png", 0),"s"])) #3
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/t1.png", 0),"t"])) #4

    #group 2 side finger up shape d i r u
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/d1.png", 0),"d"])) #5
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/i1.png", 0),"i"])) #6
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/r1.png", 0),"r"])) #7
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/u1.png", 0),"u"])) #8

    #group 3 circle type c o
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/c1.png", 0),"c"])) #9
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/c2.png", 0),"c"])) #10
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/o1.png", 0),"o"])) #11
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/o2.png", 0),"o"])) #12

    #group 4 pointing down types p q
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/p1.png", 0),"p"])) #13
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/q1.png", 0),"q"])) #14

    #group 5 many fingers up
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/b1.png", 0),"b"])) #15
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/f1.png", 0),"f"])) #16

    #group 6 sidewase point g h
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/g1.png", 0),"g"])) #17
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/h1.png", 0),"h"])) #18

    #group 7 split fingers up v k
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/v1.png", 0),"v"])) #19
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/k1.png", 0),"k"])) #20

    #group 8 unique shapes
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/e1.png", 0),"e"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/j1.png", 0),"j"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/l1.png", 0),"l"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/w1.png", 0),"w"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/x1.png", 0),"x"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/x2.png", 0),"x"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/y1.png", 0),"y"]))
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/z1.png", 0),"z"]))

    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/_1.png", 0)," "]))#space or _
    alphabetList.append(([cv2.imread("ASLproject/ASLproject/preprocessedAlphabet/bs1.png", 0),"bs"]))
    #test = "alphabetList loaded with " + str(len(alphabetList)) + " items"
    #print(test)
    #print(alphabetList[0][1])#a
    return alphabetList

def group_sorted_comparison(histROI,SobelROI,HistAlphabetList,SobelAlphabetList,distanceThreshold=0.85):
    #step 1 use HistROI to pull best shape group
    histMirrorROI = cv2.flip(histROI, 1)

    bestMatch = 0
    temp1 = 0
    temp2 = 0

    index = 0
    for testLetter in HistAlphabetList:
        temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
        temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
        if(temp1 > temp2):
            if(bestMatch < temp1):
                bestMatch = temp1
                letterGuess = testLetter[1]
        else:
            if(bestMatch < temp2):
                bestMatch = temp2
                #letterGuess = testLetter[1]
        index += 1
    print(bestMatch)
    if(bestMatch < 0.10):
        return ""

    letterGuess = ""
    if(index < 5): #fist group a s t n m
        for testLetter in SobelAlphabetList[0:5]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index > 4 and index < 9): #side finger up group d i r u
        for testLetter in SobelAlphabetList[5:9]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index > 8 and index < 13): #circle group o c
        for testLetter in SobelAlphabetList[9:13]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index == 13 or index == 14): #pointing down group p q
        for testLetter in SobelAlphabetList[13:15]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index == 15 or index == 16): #many finger up group b f
        for testLetter in SobelAlphabetList[15:17]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index == 17 or index == 18): #sideways point group g h
        for testLetter in SobelAlphabetList[17:19]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    elif(index == 19 or index == 20): #split fingers up v k
        for testLetter in SobelAlphabetList[19:21]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]
    else:
        for testLetter in SobelAlphabetList[21:]:
            temp1 = compare_images(testLetter[0],histROI,distanceThreshold)
            temp2 = compare_images(testLetter[0],histMirrorROI,distanceThreshold)
            if(temp1 > temp2):
                if(bestMatch < temp1):
                    bestMatch = temp1
                    letterGuess = testLetter[1]
            else:
                if(bestMatch < temp2):
                    bestMatch = temp2
                    letterGuess = testLetter[1]

    if(bestMatch > 0.10):
        return letterGuess
    else:
        return ""


"""
  mirrorImage = cv2.flip(image, 1)

    bestMatch = 0
    temp1 = 0
    temp2 = 0
    letterGuess = ""
    for testLetter in alphabetList:
        temp1 = compare_images(testLetter[0],image,distanceThreshold)
        temp2 = compare_images(testLetter[0],mirrorImage,distanceThreshold)
        if(temp1 > temp2):
            if(bestMatch < temp1):
                bestMatch = temp1
                letterGuess = testLetter[1]
        else:
            if(bestMatch < temp2):
                bestMatch = temp2
                letterGuess = testLetter[1]
        #test = str(bestMatch) + ", " + letterGuess
        #print(test)
    if(bestMatch > 0.10):
        return letterGuess
    else:
        return ""
        """