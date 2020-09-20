import numpy as np
import cv2
import math 


class NumberDetection:
    """
    Find signature of one digit image - for 0&5 signature50 is available 
    The signature is used for training and testing the KNN (Features) 
    """
    numberOfTrainValues = 46

    def init(self, path, passedImg = None, whichSig = 0):
        if passedImg is not None:   #there is no file to read the image, image is passed
            img = passedImg     
        else:       #we should read the image from file
            img = cv2.imread(path,0)    #image is black and white, just read it in one channel

        height, width = img.shape 
        img = self.resizing(img)    #for each digit max size is 128&...

        if img is None:
            return None

        img = cv2.bitwise_not(img)  #white word in black background
        # cv2.imshow('img',img)

        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:      #if there is no digit
            return None
        shape = contours[self.getMax(contours)]     #choose the biggest shape in the image

        center = self.getCenter(shape)      
        shape = self.makeCenterZero(shape, center)

        if whichSig == 0:  
            train_data = self.signature(shape)
        else:
            train_data = self.signature50(shape)

        train_data[self.numberOfTrainValues] = math.sqrt(height * width)

        rect = cv2.minAreaRect(shape)
        box = cv2.boxPoints(rect)

        w = math.sqrt((box.item(0) - box.item(2))**2 + (box.item(1) - box.item(3))**2)
        h = math.sqrt((box.item(2) - box.item(4))**2 + (box.item(3) - box.item(5))**2)

        x, y, w, h = cv2.boundingRect(img)
        secondHalf = img[y:4, x:x+w]     #second half of the image - money is there
        # cv2.imshow('s',secondHalf)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        train_data[self.numberOfTrainValues + 1] = w
        train_data[self.numberOfTrainValues + 2] = h
        train_data[self.numberOfTrainValues + 3] = cv2.countNonZero(secondHalf)/((4 * x) + 1)  #to remove bug between 3 & 4

        train_data = np.float32(train_data)        #train data shoud be float type
        return train_data


    def signature(self, shape):     #return signiture of the shape according to polar system - ready for training

        train_data = [0] * (self.numberOfTrainValues + 4)     #make an array with length numberOfTrainValues = 46
        train_data_temp = [0] * self.numberOfTrainValues     #make an array with length numberOfTrainValues = 46
        diff = [0] * self.numberOfTrainValues     #this array is for find max shib in second half of the image

        zarib = float(self.numberOfTrainValues) / 6.29 #2 * pi = 3.1415
        Temp_i = 0
        Temp_d = 0
        Max = 0

        for i in range(0, len(shape)):
            Temp_i = int((math.atan2(shape[i][0][1], shape[i][0][0]) + math.pi) * zarib) 
            Temp_d = self.getDistance(shape[i][0][0], shape[i][0][1])

            if Temp_d > train_data_temp[Temp_i]:    #from each range of numbers, get max
               train_data_temp[Temp_i] = float(Temp_d)

            if train_data_temp[Temp_i] > Max:     #get total max for normalizing
                Max = train_data_temp[Temp_i]

        # #for normilizing between 0 - 100
        zarib = 100 / Max
        for i in range(0, self.numberOfTrainValues):
            train_data_temp[i] = float(train_data_temp[i] * zarib)


        ##max shib are points that have the most differences between them
        ##we find max shib in the second half of the image to make differcen between 7 and 8

        # Norm : MaxFirst(!!7,8!!), NormMax
        diff[0] = train_data_temp[0] 
        Mxi = 1 
        for i in range(int(self.numberOfTrainValues/2), self.numberOfTrainValues):
            si = i % self.numberOfTrainValues
            diff[si] = train_data_temp[si] - train_data_temp[i - 1]
            z = 1
            if i > self.numberOfTrainValues * 3 / 4:
                z = 2
            if diff[si] * z > diff[Mxi]:
                Mxi = si

        for i in range(0, self.numberOfTrainValues):
            train_data[i] = train_data_temp[(i + Mxi) % self.numberOfTrainValues]

        ## jaygozin code baala - my method
        # maxIndex = self.findMaxShib(train_data_temp)
        # train_data = self.shift(train_data_temp, maxIndex)

        # train_data = np.float32(train_data)        #train data shoud be float type
        return train_data

    def signature50(self, shape):     #return signiture of the shape according to polar system - ready for training

        train_data = [999999] * (self.numberOfTrainValues + 4)     #make an array with length numberOfTrainValues = 46
        zarib = float(self.numberOfTrainValues) / 6.29       #2 * pi = 3.1415
        Temp_i = 0
        Temp_d = 0
        Max = 0

        for i in range(0, len(shape)):
            Temp_i = int((math.atan2(shape[i][0][1], shape[i][0][0]) + math.pi) * zarib);
            Temp_d = self.getDistance(shape[i][0][0], shape[i][0][1])
            if Temp_d < train_data[Temp_i]:    #from each range of numbers, get max
               train_data[Temp_i] = float(Temp_d)

        for i in range(0, (self.numberOfTrainValues + 4)):
            if train_data[i] == 999999:
                train_data[i] = 0

        for i in range(0, self.numberOfTrainValues):        
            if train_data[i] > Max:     #get total max for normalizing
                Max = train_data[i]

        ##for normilizing between 0 - 100
        zarib = 100 / Max
        for i in range(0, self.numberOfTrainValues):
            train_data[i] = float(train_data[i] * zarib)

        ##we dont change max sib in 50 signature

        # train_data = np.float32(train_data)
        return train_data

    def resizing(self,img):    #max length is 128
        
        height, width = img.shape 
        
        zarib = 0
        if height > width:
            zarib = 128 / height
        else:
            zarib = 128 / width

        h = int(height * zarib)
        w = int(width * zarib)

        if w == 0 or h == 0:
            return None

        resized = cv2.resize(img, (h, w))
        return resized

    def findMaxShib(self, train_data):    #max shib should be the first point
        diff = train_data[int(len(train_data)/2) + 1] - train_data[int(len(train_data) / 2)]
        maxVal = diff
        maxIndex = 0

        for i in range(int(len(train_data)/2) + 2, len(train_data)):
            diff = train_data[i] - train_data[i-1]
            if diff > maxVal:
                maxVal = diff
                maxIndex = i

        return maxIndex

    def shift(self, arr, point):     #shift an array according to max shib first
        newArr = [0] * len(arr)
        k = 0

        for i in range(point, len(arr)):
            newArr[k] = arr[i]
            k = k + 1
        for i in range(0, point):
            newArr[k] = arr[i]
            k = k + 1

        return newArr

    def getCenter(self,shape):    #mean of the shape is center point
        center = [0,0]

        for point in shape:
            # print(point)
            center[0] += point[0][0]
            center[1] += point[0][1]
        center[0] /= len(shape)
        center[1] /= len(shape)

        center[0] = int(center[0])
        center[1] = int(center[1])

        # print(center)
        return center

    def makeCenterZero(self, shape, center):    #reduce center from all pixel, so center is zero now
        for point in shape:
            point[0][0] -= center[0]
            point[0][1] -= center[1]
        return shape

    def getMax(self, contours):    #return biggest shape in the image - help to ignore noise
        maxSize = 0
        index = 0
        for i in range(0, len(contours)):
            if(len(contours[i]) > maxSize):
                maxSize = len(contours[i])
                index = i
        return int(index)

    def getDistance(self, x, y):    #because center is (0,0), distant to center is sqrt(x^2 + y^2)
        return math.sqrt(x**2 + y**2)
