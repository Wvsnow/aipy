# -*- coding: utf-8 -*-

from sklearn import neighbors
from knn.data_util import DataUtils
import datetime
import numpy
import traceback


def main():
    trainfile_X = '../data/dataset/MNIST/train-images.idx3-ubyte'
    trainfile_y = '../data/dataset/MNIST/train-labels.idx1-ubyte'
    testfile_X = '../data/dataset/MNIST/t10k-images.idx3-ubyte'
    testfile_y = '../data/dataset/MNIST/t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(filename=testfile_X).getImage()
    test_y = DataUtils(filename=testfile_y).getLabel()

    return train_X, train_y, test_X, test_y


def load_npz_data(path):
    """
    The function to load data-set
    :param path: path of data
    :return: data detail, train samples 60000, test samples 10000, and X with dimension 28*28, Y with dimension 1
    """
    files = numpy.load(path)
    x_train, y_train = files['x_train'], files['y_train']
    x_test, y_test = files['x_test'], files['y_test']
    files.close()
    return x_train, y_train, x_test, y_test


def testKNN4MnistDataSet():
    """
    To test KNN with Mnist-DataSet
    :return: Test Result of KNN for Mnist-DataSet
    """
    startTime = datetime.datetime.now()
    try:
        train_X, train_y, test_X, test_y = main()
        # train_X, train_y, test_X, test_y = load_npz_data(r'D:\data\ai\dataset\ocr\mnist.npz')

        knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_X, train_y)

        testCount = len(test_y)
        print("Sum {} test sample".format(testCount))
        match = 0
        for i in range(testCount):
            predictLabel = knn.predict(test_X[i].reshape((1, -1)))[0]
            if (0 == i % 400):
                print('index {}, test {},predict {}'.format(i, test_y[i], predictLabel))
            if (predictLabel == test_y[i]):
                match += 1
    except Exception as ex:
        print("Exception")
        traceback.print_exc()
    finally:
        endTime = datetime.datetime.now()
        # use time: 0:15: 45.300228
        # error rate: 0.039000000000000035
        print('use time: ' + str(endTime - startTime))
        print('error rate: ' + str(1 - (match * 1.0 / len(test_y))))
        print("KNN end")


def testDNN4MnistDataSet():
    print("DNN end")


if __name__ == "__main__":
    testKNN4MnistDataSet()
