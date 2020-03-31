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

#referencing https://sandipanweb.wordpress.com/2017/10/22/feature-detection-with-harris-corner-detector-and-matching-images-with-feature-descriptors-in-python/
def get_descriptor(I, X, Y):
    pass

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

def abs_distance_matching(feature1, feature2, threshold=50):
    #this function is used to compute the sum of absolute distance between features
    #Matching features should be in the same aproximate locations
    pass

def compute_matches(image1, image2):
    #used to compute matches between images. return highest fedelity match
    pass