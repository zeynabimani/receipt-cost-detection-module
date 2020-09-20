import numpy as np
import cv2
import pickle
from NumberDetection import NumberDetection


class KNN:
    """
    Working by Knn, test method checks your data acooriding to signature that is given to it
    """

    counter = 0
    knn = cv2.ml.KNearest_create()     
    knn50 = cv2.ml.KNearest_create()  #we use differen knn with different train data for 0&5 - signature50
    K = 9

    def init(self):
        train_data = self.loadTrain_data()
        train_data50 = self.loadTrain_data50()

        train_labels = self.loadTrain_labels()
        train_labels50 = self.loadTrain_labels50()

        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        self.knn50.train(train_data50, cv2.ml.ROW_SAMPLE, train_labels50)


    def test(self, signature, signature50):

        if signature is None or signature50 is None:   #this is when the size of image is 0
            return -1

        signature2 = np.reshape(signature, (-1, 50))

        ret, results, neighbours ,dist = self.knn.findNearest(signature2, self.K)

        res = -1

        if int(results[0][0]) == 50:
            res = self.test50(signature50)
        else:
            res = int(results[0][0])

        return res

    def test50(self, signature50):
        signature2 = np.reshape(signature50, (-1, 50))
        ret, results, neighbours ,dist = self.knn50.findNearest(signature2, self.K)

        return int(results[0][0])

    def loadTrain_data(self):
        filename = 'pre-trained/train_data'
        infile = open(filename,'rb')
        train_data = pickle.load(infile)
        infile.close()
        return train_data

    def loadTrain_data50(self):
        filename = 'pre-trained/train_data50'
        infile = open(filename,'rb')
        train_data50 = pickle.load(infile)
        infile.close()
        return train_data50
        
    def loadTrain_labels(self):
        filename = 'pre-trained/train_labels'
        infile = open(filename,'rb')
        loadtrain_labels = pickle.load(infile)
        infile.close()
        return loadtrain_labels
        
    def loadTrain_labels50(self):
        filename = 'pre-trained/train_labels50'
        infile = open(filename,'rb')
        loadtrain_labels50 = pickle.load(infile)
        infile.close()
        return loadtrain_labels50


