import numpy as np
import cv2
import math 
import random
import string
import datetime
from NumberDetection import NumberDetection
from knn import KNN
from detect import detect


class prepare:
    """
    Prepareing the whole image - preprocess and then check the date and cost
    """

    counter = 29  #variables in class body consider as static
    nd = NumberDetection()
    myKnn = KNN()
    myKnn.init()
    det = detect()
    
    def init(self, path, passedImg = None):
        """
        You can pass either the image Mat by some other module or read the image from a file

        Do some preprocess on the image and check for the date and cost by with the help of the Detect class 
        """

        if passedImg is not None:
            img = passedImg
        else:
            img = cv2.imread(path)

        ready = self.preprocess(img)
        # cv2.imshow('ready', ready)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  
        # self.counter = self.counter + 1
        # cv2.imwrite(str(self.counter) + ".png", ready)
        x, y, w, h = cv2.boundingRect(ready)

        firstHalf = ready[int(y+h/6):int(y+3*h/4), x:x+w]
        date = self.det.searchDate(firstHalf)

        secondHalf = ready[int(y+h/2):y+h, x:x+w]
        money = self.det.searchMoney(secondHalf)

        DateAndCost = []
        DateAndCost.append(date)
        DateAndCost.append(money)
        print(DateAndCost)
        return DateAndCost


    def resizing(self,img):    #max length is 720 - acoording to quality of currunt images
        height, width = img.shape 
        zarib = 0
        if height > width:
            zarib = 720 / height
        else:
            zarib = 720 / width
        resized = cv2.resize(img, (int(height * zarib), int(width * zarib)))
        return resized

    def preprocess(self, img):
        ##sharp filter (become more clear)
        blur = cv2.GaussianBlur(img,(0, 0), 3) 
        sharp = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

        gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
        gray = self.resizing(gray)
        
        ##GaussianBlur to remove noises
        blur = cv2.GaussianBlur(gray, (3,3), 1)
        
        ##binarization
        gaus = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 7)
        (thresh, blackAndWhiteImage) = cv2.threshold(gaus, 127, 255, cv2.THRESH_BINARY)   #dark part become black - i'm not sure actully

        return blackAndWhiteImage


if __name__ == '__main__':
    prep = prepare()
    help(prep)
    for i in range(12, 13):
        prep.init("dataset/testData-crop/" + str(i) + ".jpg")
        