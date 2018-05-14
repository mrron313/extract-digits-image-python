from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np
import imutils
from imutils import contours
from cv2 import imwrite

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

img = cv.imread('images/1.png',0)

rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 8))
sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, rectKernel)

gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0,
    ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
thresh = cv.threshold(gradX, 0, 255,
    cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
 
# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []

# loop over the contours
for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
        # contours can further be pruned on minimum/maximum width
        # and height
        #cv.rectangle(img,(x,y),(x+w,y+h),(125,0,200),5)
        crop_img = img[y-5:y+h+5,x-5:x+w+5]

        th2 = cv.threshold(crop_img, 0, 255,
                           cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
                           
        digitCnts = cv.findContours(th2.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
        
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]                   
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        for d in digitCnts:
            (x, y, w, h) = cv.boundingRect(d)
            roi = crop_img[y-1 : y+h+1 , x-1 : x+w+1]
            roi = cv.resize(roi, (200,120))
            
            th3 = cv.threshold(roi, 0, 255,
                           cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
                           
            arifs = cv.findContours(th3.copy(), cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)               
            
            arifs = arifs[0] if imutils.is_cv2() else arifs[1]

            arif = max(arifs, key = lambda x: cv.contourArea(x))

            drawing2 = np.zeros(th3.shape,np.uint8)
            
            cv.fillPoly(drawing2, pts =[arif], color=(255,255,255))
            
            main_digit = cv.bitwise_and(th3,th3, mask=drawing2)

            main_threshold = cv.threshold(main_digit, 0, 255,
                           cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            
            
            cv.imshow('main', main_threshold)
            cv.waitKey(0)
            cv.destroyAllWindows()


