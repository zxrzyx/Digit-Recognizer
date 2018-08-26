from sklearn import tree, svm
import matplotlib.pyplot as plt
import torch
import csv
import numpy as np

n_labels = []
n_train_images = []
n_train_data = []
n_test_images = []
n_test_data = []
n_test_labels = []

tensor = torch.abs(torch.FloatTensor(-1))

def read_train_csv(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            n_labels.append(row[0])
            arr = np.array(row[1:], dtype=int)
            n_train_data.append(arr)
            n_train_images.append(arr.reshape(28, 28))
    # print(n_labels)
    # print(n_images[1])
    # plt.axis('off')
    # plt.imshow(n_images[1])
    # plt.show()
    print('read train')
    print(tensor)

def read_test_csv(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            arr = np.array(row, dtype=int)
            n_test_data.append(arr)
            n_test_images.append(arr.reshape(28, 28))
    print('read test')

def analyzing():
    print('analyzing')
    classifier = svm.SVC(kernel='poly')
    classifier.fit(n_train_data, n_labels)

    predict = classifier.predict(n_test_data)
    for index in range(len(predict)):
        n_test_labels.append([index, predict[index]])




def print_result():
    print('print')
    with open('result.csv', 'w', newline='') as csvfile:
        csvfile.write('ImageId,Label\n')
        writer = csv.writer(csvfile)
        for row in n_test_labels:
            writer.writerow(row)



if __name__ == "__main__":
    read_train_csv('all/train.csv')
    read_test_csv('all/test.csv')
    analyzing()
    print_result()

