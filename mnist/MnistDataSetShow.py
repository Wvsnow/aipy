# !/D:/Applications/python/python3.5.4/python
"""
This function to test ans show the dataset of mnist.npz
To show some one handwriting
"""
import matplotlib.pyplot as plt
import numpy
import random


def load_data(path):
    """
    The function to load data-set
    :param path: path of data
    :return: data detail
    """
    files = numpy.load(path)
    x_train, y_train = files['x_train'], files['y_train']
    x_test, y_test = files['x_test'], files['y_test']
    files.close()
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    """
    The main function.
    Test data-set of mnist for OCR
    """
    # region The body of function
    path = '../data/dataset/MNIST/mnist.npz'
    x_train, y_train, x_test, y_test = load_data(path)

    showCount = 3
    for i in range(showCount):
        index = random.randint(1, 60000)
        plt.imshow(x_train[index], plt.get_cmap('PuBuGn_r'))
        plt.show()
    # endregion
