
#imports
#import pyttsx3 #USE VERSION 2.71
import numpy as np
from matplotlib import pyplot as plt
import cv2 #openCV
#import regionOfInterest
#import resize
import imageProcesses



"""
img = image1

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print(kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
plt.imshow(img2)
plt.show()
"""
"""
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,flags=2)

plt.imshow(img3),plt.show()


"""
image1 = cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0)
image2 = cv2.imread("ASLproject/ASLproject/edgePreprocess/x2.png", 0)

#cv2.ORB
orb = cv2.ORB_create()

#image = cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0)
#image2 = cv2.imread("ASLproject/ASLproject/edgePreprocess/x1.png", 0)
print("step0")
#cv2.imshow("test",image) 
#cv2.waitKey(0)

kp1 = orb.detect(image1,None)
kp2 = orb.detect(image2,None)
print("step1")
kp1, des1 = orb.compute(image1, kp1)
kp2, des2 = orb.compute(image2, kp2)
print("step2")
img1 = cv2.drawKeypoints(image1, kp1, image1, color=(0,255,0), flags=0)
img2 = cv2.drawKeypoints(image2, kp2, image2, color=(0,255,0), flags=0)
print("step3")

#numpy_horizontal = np.hstack((img1, img2))
#cv2.imshow("test",numpy_horizontal)
#cv

cv2.imshow("test",img1)
cv2.waitKey(0)
cv2.imshow("test",img2)
cv2.waitKey(0)



# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)




# Apply ratio test
good = []
bad = []
for m,n in matches:
    if(m.distance < 0.85*n.distance):#.75*n
        good.append([m])
    else:
        bad.append([m])

print(len(good))
print(len(bad))
percent_match = len(good)/(len(good)+len(bad))
print(percent_match)


# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = image1
#img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,image1,flags=2)
#cv2.imshow("testout",img3)

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
