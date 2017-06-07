import cv2
import sys
import numpy as np

from scipy.io import loadmat

def convert():
    labels = loadmat('tmp/data/devkit/cars_meta.mat')
    car_labels = []
    for label in labels['class_names'][0]:
        car_labels.append(label[0])

    labels_file = open("tmp/data/devkit/car_labels.txt", "w")
    labels_file.write("\n".join(car_labels))
    labels_file.close()

if __name__ == '__main__':
    convert()
