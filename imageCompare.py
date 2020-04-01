import numpy as np
import cv2 #openCV
import resize

alphabet_list = None

#This function pulls preproccessed images for use in comparison
def initialize_comparison_library():
    #abc... and bs(backspace) and space
    alphabetList = []
    alphabetList.append([cv2.imread("preprocessed alphabet/a1.png"), "a"])
    alphabetList.append([cv2.imread("preprocessed alphabet/b1.png"), "b"])

    alphabetList.append([cv2.imread("preprocessed alphabet/c1.png"), "c"])
    alphabetList.append([cv2.imread("preprocessed alphabet/c2.png"), "c"])

    alphabetList.append([cv2.imread("preprocessed alphabet/d1.png"), "d"])
    alphabetList.append([cv2.imread("preprocessed alphabet/e1.png"), "e"])
    alphabetList.append([cv2.imread("preprocessed alphabet/f1.png"), "f"])
    alphabetList.append([cv2.imread("preprocessed alphabet/g1.png"), "g"])
    alphabetList.append([cv2.imread("preprocessed alphabet/h1.png"), "h"])
    alphabetList.append([cv2.imread("preprocessed alphabet/i1.png"), "i"])
    alphabetList.append([cv2.imread("preprocessed alphabet/j1.png"), "j"])
    alphabetList.append([cv2.imread("preprocessed alphabet/k1.png"), "k"])
    alphabetList.append([cv2.imread("preprocessed alphabet/l1.png"), "l"])
    alphabetList.append([cv2.imread("preprocessed alphabet/m1.png"), "m"])
    alphabetList.append([cv2.imread("preprocessed alphabet/n1.png"), "n"])

    alphabetList.append([cv2.imread("preprocessed alphabet/o1.png"), "o"])
    alphabetList.append([cv2.imread("preprocessed alphabet/o2.png"), "o"])

    alphabetList.append([cv2.imread("preprocessed alphabet/p1.png"), "p"])
    alphabetList.append([cv2.imread("preprocessed alphabet/q1.png"), "q"])
    alphabetList.append([cv2.imread("preprocessed alphabet/r1.png"), "r"])
    alphabetList.append([cv2.imread("preprocessed alphabet/s1.png"), "s"])
    alphabetList.append([cv2.imread("preprocessed alphabet/t1.png"), "t"])
    alphabetList.append([cv2.imread("preprocessed alphabet/u1.png"), "u"])
    alphabetList.append([cv2.imread("preprocessed alphabet/v1.png"), "v"])
    alphabetList.append([cv2.imread("preprocessed alphabet/w1.png"), "w"])

    alphabetList.append([cv2.imread("preprocessed alphabet/x1.png"), "x"])
    alphabetList.append([cv2.imread("preprocessed alphabet/x1.png"), "x"])

    alphabetList.append([cv2.imread("preprocessed alphabet/y1.png"), "y"])
    alphabetList.append([cv2.imread("preprocessed alphabet/z1.png"), "z"])

    alphabetList.append([cv2.imread("preprocessed alphabet/_1.png"), "_"])#space or _
    alphabetList.append([cv2.imread("preprocessed alphabet/bs1.png"), "bs"])
    #test = "alphabetList loaded with " + str(len(alphabetList)) + " items"
    #print(test)
    #print(alphabetList[0][1])#a
    return alphabetList

def compareImages(source, template, method):
    # Ensure both pictures are gray scale
    try:
        if source.shape[2]:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    except:
        pass

    try:
        if template.shape[2]:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    except:
        pass
    

    # Resize both pictures to same scale
    source = resize.normalize_image_size(source)
    template = resize.normalize_image_size(template)

    # Find matches between pictures
    contours, heirarchy = cv2.findContours(source, 2, 1)
    cnt1 = contours[0]
    contours, heirarchy = cv2.findContours(template, 2, 1)
    cnt2 = contours[0]

    ret = cv2.matchShapes(cnt1, cnt2, method, 0.0)

    return ret

def compareToLibrary(img):
    # Initialize our library of templates
    global alphabet_list
    if (alphabet_list is None):
        print("Initializing comparison library...")
        alphabet_list = initialize_comparison_library()

    # print(alphabet_list)

    best_match_percent = 0.0
    best_match_letter = None

    # For each template, find match percentage with source image
    for i in np.arange(0, len(alphabet_list)):
        template = alphabet_list[i][0]
        letter = alphabet_list[i][1]
        # cv2.imshow("template", template)
        # print(letter)
        # cv2.waitKey(0)

        match = compareImages(img, template, 2)
        percentage = ((1.0 - match) * 100)

        # Keep highest one and its letter
        if (percentage > best_match_percent):
            best_match_percent = percentage
            best_match_letter = letter

    # Return letter and matching percentage
    return best_match_letter, best_match_percent

def main():
    # For testing purposes
    print("Part 1")
    source = cv2.imread("roi.png")
    template = cv2.imread("preprocessed alphabet/_1.png")
    total = 0.0
    items = 0

    percentage = compareImages(source, template, 1)
    if (percentage > 1.0):
        print("1: No match")
    else:
        total += percentage
        items += 1
        print(f"1: {((1.0 - percentage) * 100):0.2f} % match")

    percentage = compareImages(source, template, 2)
    if (percentage > 1.0):
        print("2: No match")
    else:
        total += percentage
        items += 1
        print(f"2: {((1.0 - percentage) * 100):0.2f} % match")

    percentage = compareImages(source, template, 3)
    if (percentage > 1.0):
        print("3: No match")
    else:
        total += percentage
        items += 1
        print(f"3: {((1.0 - percentage) * 100):0.2f} % match")

    if (items > 0 and total > 0.0):
        average = total / items
        print(f"Average: {((1.0 - average) * 100):0.2f} % match")

    print("Part 2")
    letter, percent = compareToLibrary(source)

    if (letter != None):
        print(f"Best match is '{letter}' at {percent:0.2f} %")
    else:
        print("There was no match in the whole library.")

    # cv2.imshow("Result", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#run main
if __name__ == "__main__":
    main()