import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from NumberDetection import NumberDetection

nd = NumberDetection()

train_data = nd.init("dataset/db/0/0.png")
train_labels = [50.0]
# plt.plot(train_data)

for i in range(1, 728): #0 P
    signature = nd.init("dataset/db/0/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [50.0]))

for i in range(0, 400): #0 K
    signature = nd.init("dataset/new_db/0/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [50.0]))


for i in range(0, 390): #5 P
    signature = nd.init("dataset/db/5/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [50.0]))

for i in range(0, 510): #5 K
    signature = nd.init("dataset/new_db/5/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [50.0]))


for i in range(0, 1031): #1 K
    signature = nd.init("dataset/db/1/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [1.0]))

for i in range(0, 940): #1 P
    signature = nd.init("dataset/new_db/1/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [1.0]))


for i in range(0, 547): #2 K
    signature = nd.init("dataset/db/2/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [2.0]))

for i in range(0, 651): #2 P
    signature = nd.init("dataset/new_db/2/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [2.0]))


for i in range(0, 582): #3 P
    signature = nd.init("dataset/db/3/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [3.0]))

for i in range(0, 673): #3 K
    signature = nd.init("dataset/new_db/3/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [3.0]))


for i in range(0, 354): #4 P
    signature = nd.init("dataset/db/4/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [4.0]))

for i in range(0, 380): #4 K
    signature = nd.init("dataset/new_db/4/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [4.0]))


for i in range(0, 422): #6 P
    signature = nd.init("dataset/db/6/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [6.0]))

for i in range(0, 450): #6 K
    signature = nd.init("dataset/new_db/6/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [6.0]))

for i in range(0, 321): #7 P
    signature = nd.init("dataset/db/7/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [7.0]))

for i in range(0, 433): #7 K
    signature = nd.init("dataset/new_db/7/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [7.0]))


for i in range(0, 432): #8 P
    signature = nd.init("dataset/db/8/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [8.0]))

for i in range(1, 539): #8 K
    signature = nd.init("dataset/new_db/8/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [8.0]))


for i in range(0, 448): #9 P
    signature = nd.init("dataset/db/9/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [9.0]))

for i in range(0, 524): #9 K
    signature = nd.init("dataset/new_db/9/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [9.0]))


for i in range(0, 236): #momayez P
    signature = nd.init("dataset/db/Momayez/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [10.0]))

for i in range(0, 304): #momayez K
    signature = nd.init("dataset/new_db/momayez/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [10.0]))


for i in range(0, 59): #L P
    signature = nd.init("dataset/db/L/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [11.0]))

for i in range(0, 73): #L K
    signature = nd.init("dataset/new_db/L/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [11.0]))


for i in range(0, 293): #R P
    signature = nd.init("dataset/db/R/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [12.0]))

for i in range(0, 297): #R K
    signature = nd.init("dataset/new_db/R/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [12.0]))


for i in range(0, 229): #ya ALL
    signature = nd.init("dataset/db/Ya/" + str(i) + ".png")
    # plt.plot(signature)
    train_data = np.vstack((train_data, signature))
    train_labels = np.vstack((train_labels, [13.0]))


train_data = np.float32(train_data)
train_labels = np.float32(train_labels)

filename = 'train_data'
outfile = open(filename,'wb')
pickle.dump(train_data,outfile)
outfile.close()

filename = 'train_labels'
outfile = open(filename,'wb')
pickle.dump(train_labels,outfile)
outfile.close()

#-------------------------------------------------
###seond part - 50


# nd = NumberDetection()

# train_data = nd.init("", cv2.imread("dataset/db/0/0.png", 0), 50)
# train_labels = [0.0]


# for i in range(1, 728): #0 P
#     signature = nd.init("", cv2.imread("dataset/db/0/" + str(i) + ".png", 0), 50)
#     # plt.plot(signature)
#     train_data = np.vstack((train_data, signature))
#     train_labels = np.vstack((train_labels, [0.0]))

# for i in range(1, 400): #0 K
#     signature = nd.init("", cv2.imread("dataset/new_db/0/" + str(i) + ".png", 0), 50)
#     # plt.plot(signature)
#     train_data = np.vstack((train_data, signature))
#     train_labels = np.vstack((train_labels, [0.0]))


# for i in range(0, 390): #5 P
#     signature = nd.init("", cv2.imread("dataset/db/5/" + str(i) + ".png", 0), 50)
#     # plt.plot(signature)
#     train_data = np.vstack((train_data, signature))
#     train_labels = np.vstack((train_labels, [5.0]))

# for i in range(0, 510): #5 K
#     signature = nd.init("", cv2.imread("dataset/new_db/5/" + str(i) + ".png", 0), 50)
#     # plt.plot(signature)
#     train_data = np.vstack((train_data, signature))
#     train_labels = np.vstack((train_labels, [5.0]))


# train_data = np.float32(train_data)
# train_labels = np.float32(train_labels)

# filename = 'train_data50'
# outfile = open(filename,'wb')
# pickle.dump(train_data,outfile)
# outfile.close()

# filename = 'train_labels50'
# outfile = open(filename,'wb')
# pickle.dump(train_labels,outfile)
# outfile.close()

# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()