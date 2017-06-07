import cv2
import sys
import numpy as np

import cifar10

from scipy.io import loadmat


def convert(type):
    output_file = open('data/cifar10_data/cifar-10-batches-bin/' + type +'_batch.bin', 'ab')
    car_annos = loadmat('data/devkit/cars_' + type + '_annos.mat')
    print car_annos
    for annotations in car_annos['annotations'][0]:
        print annotations
        real_label = int(annotations[4].item())

        label = 0
        if 2 <= real_label <= 7:
            label = 1
        elif 12 <= real_label <= 25:
            label = 2
        elif 26 <= real_label <= 38:
            label = 3
        elif 39 <= real_label <= 44:
            label = 4
        elif 54 <= real_label <= 75:
            label = 5
        elif 76 <= real_label <= 81:
            label = 6
        elif 83 <= real_label <= 97:
            label = 7
        elif 101 <= real_label < 104:
            label = 8
        elif 106 <= real_label < 117:
            label = 9
        elif 118 <= real_label < 122:
            label = 10
        elif 126 <= real_label < 129:
            label = 11
        elif 130 <= real_label < 140:
            label = 12
        elif 145 <= real_label < 149:
            label = 13
        elif 150 <= real_label < 153:
            label = 14
        elif 161 <= real_label < 166:
            label = 15
        elif 168 <= real_label < 171:
            label = 16
        elif 175 <= real_label < 177:
            label = 17
        elif 181 <= real_label < 184:
            label = 18
        elif 186 <= real_label < 189:
            label = 19

        img_path = 'data/cars_' + type + '/' + str(annotations[5].item())
        print img_path
        im = cv2.imread(img_path)
        im = cv2.resize(im, (32, 32))

        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        output = np.array([label] + list(r) + list(g) + list(b), dtype = np.uint8)
        # output = np.array([label] + list(im.flatten()), dtype = np.uint8)
        output.tofile(output_file)

    output_file.close()

if __name__ == '__main__':
    convert('train')
    convert('test')


