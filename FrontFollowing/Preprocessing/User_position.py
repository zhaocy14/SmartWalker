#-*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from typing import List, Tuple, NoReturn


class User_Postition_Estimate(object):
    def __init__(self, img: np.ndarray):
        """raw data and data shape, and expected positional information of user"""
        self.image = img
        self.width = 32
        self.height = 24
        """some processing parameter"""
        self.bias_for_threshold = 2.21
        self.user_x = 0
        self.user_y = 0

    def get_new_img(self, img: np.ndarray):
        self.image = img
        return None

    def binarization(self):
        """according to an average value of the image to decide the threshold"""
        img = np.copy(self.image)
        if len(img.shape) == 2:
            threshold = img.mean() + self.bias_for_threshold
            img[img < threshold] = 0
            img[img >= threshold] = 1
        elif len(img.shape) == 3:
            """suppose read from a jpg file with three channels"""
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, img = cv.threshold(img, 127, 255, 0)
        return img

    def filter(self, img):
        img_new = np.copy(img)
        filter_kernel = np.ones((2, 2)) / 4
        """other filters"""
        # filter_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/10
        # filter_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        for j in range(1):
            img_new = cv.filter2D(img_new, -1, filter_kernel)
        return img_new

    def get_COM(self,show:bool = True) -> NoReturn:
        """get the center of mass of the image"""
        img = self.binarization()
        filter_times = 2
        for i in range(filter_times):
            """erase the checkerboard effect"""
            img = self.filter(img)
        img, contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        """select main part"""
        list_area = []
        for i in contours:
            list_area.append(cv.contourArea(i))
        contour_dict = dict(zip(list_area,contours))
        contour_dict = sorted(contour_dict.items(), key=lambda k: k[0], reverse=True)
        contours = []
        if len(contour_dict) == 0:
            self.user_x , self.user_y = 0, 0
        elif len(contour_dict) == 1:
            contours.append(contour_dict[0][1])
        else:
            if abs(contour_dict[0][0]-contour_dict[1][0]) <= 10000000000:
                contours.append(contour_dict[0][1])
                contours.append(contour_dict[1][1])
            else:
                contours.append(contour_dict[0][1])

        COM = []
        if len(contours) != 0:
            for i in contours:
                contours_mom = cv.moments(i)
                COM_i = (int(contours_mom['m10'] / contours_mom['m00']), int(contours_mom['m01'] / contours_mom['m00']))
                COM.append(COM_i)

        if len(COM) == 1:
            self.user_x = COM[0][0]
            self.user_y = COM[0][1]
        elif len(COM) == 2:
            self.user_x = (COM[0][0] + COM[1][0])/2
            self.user_y = (COM[0][1] + COM[1][1])/2

        if show:
            show_img = np.copy(self.image)
            show_img = cv.drawContours(show_img, contours, -1, (0,0,255), 4)
            for i in COM:
                cv.circle(show_img, i, 2, (0, 0, 255), 2)
            cv.imshow('drawimg', show_img)
            cv.waitKey(0)


if __name__ == '__main__':
    img = cv.imread('Record_data/img362.jpg')
    UP = User_Postition_Estimate(img)
    UP.get_COM(False)