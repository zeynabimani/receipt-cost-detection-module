import numpy as np
import cv2
import math 
from NumberDetection import NumberDetection
from knn import KNN


class detect:
    """
    Search for the date and cost
    """

    nd = NumberDetection()
    myKnn = KNN()
    myKnn.init()


    def searchDate(self, firstHalf):
        
        x, y, w, h = cv2.boundingRect(firstHalf)
        whiteCnt = cv2.bitwise_not(firstHalf)     #white word in dark background

        ##find sum of each line, so we can find words -> higher sum (more white pixels)
        sums = [0] * h
        for i in range(0, h):
            sums[i] = sum(whiteCnt[i]) / float(w + 1)   #percent

        ##according to sums we find number of lines
        first = 0
        lines = []
        sw = 1
        for i in range(h - 2, -1, -1):
            if sw == 1 and sums[i] <= 15: #15 percent
                sw = 1

            elif sw == 1 and sums[i] > 15:
                sw = 2

            elif sw == 2 and sums[i] <= 15:
                lines.append(i)
                sw = 1
        if len(lines) == 0:
            return []

        allCntRes = [{} for _ in range(len(lines))]  
        allDATE = [0] * len(lines)   

        contours, _ = cv2.findContours(whiteCnt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) > 35:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = firstHalf[y:y+h, x:x+w]

                signature = self.nd.init("", roi)
                signature50 = self.nd.init("", roi, 50)
                res = self.myKnn.test(signature, signature50)

                index = 0
                centerY = y + int(h/2)
                for i in range(0, len(lines)):
                    if centerY > lines[i]:
                        index = i
                        break
                            
                allCntRes[index][x] = res
                if res == 10: # /
                    allDATE[index] = allDATE[index] + 1

        for i in range(0, len(lines)):   #we search for date from top to button
            sw = 0
            firstIndex = 0
            if allDATE[i] == 2: #it might be date
                digs = list(dict(sorted(allCntRes[i].items())).values())

                if len(digs) < 8:   #it cant be date
                    continue
                for i in range(0, len(digs)):
                    if i >= 2 and digs[i] == 10 and sw == 0:
                        sw = 1
                        firstIndex = i

                    elif digs[i] == 10 and sw == 1:

                        if (i - firstIndex) == 3: # date?
                            if len(digs) - i >= 2:
                                return digs[(firstIndex - 2):(i + 3)]
                            else:
                                return []
                        elif (i - first) < 3:
                            sw = 0
                        else:
                            sw = 1
                            firstIndex = i

        return []    

       
    def searchMoney(self, secondHalf):
        x, y, w, h = cv2.boundingRect(secondHalf)
        whiteCnt = cv2.bitwise_not(secondHalf)     #white word in dark background

        ##find sum of each line, so we can find words -> higher sum (more white pixels)
        sums = [0] * h
        for i in range(0, h):
            sums[i] = sum(whiteCnt[i]) / float(w + 1)   #percent

        ##according to sums we find number of lines
        first = 0
        lines = []
        sw = 1
        for i in range(h - 2, -1, -1):
            if sw == 1 and sums[i] <= 20: #20 percent
                sw = 1

            elif sw == 1 and sums[i] > 20:
                sw = 2

            elif sw == 2 and sums[i] <= 20:
                lines.append(i)
                sw = 1
        if len(lines) == 0:
            return []

        allCntRes = [{} for _ in range(len(lines))]  
        allCntY = [{} for _ in range(len(lines))]   
        allCntVal = [{} for _ in range(len(lines))]   
        allRIAL = [[0,0,0] for _ in range(len(lines))]   

        contours, _ = cv2.findContours(whiteCnt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) > 35:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = secondHalf[y:y+h, x:x+w]

                signature = self.nd.init("", roi)
                signature50 = self.nd.init("", roi, 50)
                res = self.myKnn.test(signature, signature50)

                # if res < 10:
                #     cv2.putText(secondHalf, str(res), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                # elif res == 10:
                #     cv2.putText(secondHalf, "/", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                # elif res == 11:
                #     cv2.putText(secondHalf, "L", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                # elif res == 12:
                #     cv2.putText(secondHalf, "R", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                # elif res == 13:
                #     cv2.putText(secondHalf, "Y", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                # elif res == 14:
                #     cv2.putText(secondHalf, ":", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)

                index = 0
                centerY = y + int(h/2)
                for i in range(0, len(lines)):
                    if centerY > lines[i]:
                        index = i
                        break
                            
                allCntRes[index][x] = res
                allCntY[index][x] = centerY
                allCntVal[index][x] = cv2.countNonZero(roi)
                if res >= 11 and res <= 13: #R - Ya _ L
                    allRIAL[index][res - 11] = 1

        # cv2.imshow('closing',secondHalf)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        sw = 0  #it become 1, when we find money
        money = []
        for i in range(0, len(lines)):   #we search for money from button to top
            if sw == 1: 
                break

            valid = 0
            for j in range(0, 3):
                valid += allRIAL[i][j]
                if valid >= 2:  #it contains RIAL alphabet

                    digs = list(dict(sorted(allCntRes[i].items())).values())
                    ys = list(dict(sorted(allCntY[i].items())).values())
                    vals = list(dict(sorted(allCntVal[i].items())).values())
                    
                    isRial = 0
                    if len(digs) <= 3:  #insufficient digits
                        break
                    for i in range(0, 4):
                        if digs[i] >= 11 and digs[i] <= 13:   #RIAL alphabet should be at te beginning
                            isRial = isRial + 1
                    zero = 0
                    if isRial >= 2:    #We find it - go out after that
                        for i in range(2, len(digs)):
                            if digs[i] == 0:
                                zero = zero + 1
                            if zero >= 3 and digs[i] != 0:
                                break  #because it consider MABLAGH as a number, we think that money should end with at least 000
                            elif zero >= 2 and digs[i] == 0:
                                if vals[i] <= 50:
                                    break  #because it consider : as 0, centers of : & 0 are different in number of white pixels 

                            if digs[i] < 10:
                                if(ys[i] - ys[i - 1]) < 8:    #comma detects as a number, comma is lower than our digits
                                    money.append(digs[i])
                        sw = 1
                        break
                    break

        return money
