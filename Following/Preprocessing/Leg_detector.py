import numpy as np
import math
import cv2
from PIL import Image
from rplidar import RPLidar
import time
from sklearn.cluster import KMeans
import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")

class Leg_detector(object):

    def __init__(self, portal: str = '/dev/ttyUSB4', is_show:bool=False):
        self.rplidar = RPLidar(portal)  # '/dev/ttyUSB1'
        self.kmeans = KMeans(n_clusters=2)
        self.left_leg = np.zeros((1, 2))
        self.right_leg = np.zeros((1, 2))

        self.scope = 1
        self.size = 300
        self.half_size = int(self.size / 2)
        self.column_boundry = self.half_size-20
        self.filter_theta = 150
        self.bottom_boundary = self.half_size-100

        self.center_point = np.array([self.half_size+45,self.half_size])
        self.is_show = is_show

    def turn_to_img(self, original_list: list, show: bool = False):
        """turn the scan data list into a ndarray"""
        img = np.zeros((self.size, self.size))
        for i in range(len(original_list)):
            theta = original_list[i][1]
            theta = -theta / 180 * math.pi
            distance = original_list[i][2] / 10  # unit: mm
            index_x = int(distance * math.cos(theta) + self.half_size)
            index_y = int(distance * math.sin(theta) + self.half_size)
            # print(index_x,index_y)
            index_x = min(max(index_x, 0), self.size - 1)
            index_y = min(max(index_y, 0), self.size - 1)
            img[index_x, index_y] = 1
        if show:
            im = np.copy(img)
            im[self.half_size - 1:self.half_size + 1, self.half_size - 1:self.half_size + 1] = 1
            size = int(self.size * self.scope)
            im = Image.fromarray(im)
            im = im.resize((size, size), Image.BILINEAR)
            im = np.array(im)
            cv2.imshow("laser", im)
            cv2.waitKey(1)
        return img

    def detect_leg(self, kmeans: KMeans, img: np.ndarray, theta: float = 160, show: bool = False):
        theta = theta / 180 * math.pi
        tan_theta = math.tan(theta / 2)
        im = np.copy(img)
        im[0:self.half_size, :] = 0
        column_boundry = self.column_boundry
        im[:, 0:column_boundry] = 0
        im[:, self.size - column_boundry:self.size] = 0
        row_boundary = self.bottom_boundary
        im[self.size - row_boundary:self.size, :] = 0
        for i in range(self.half_size, self.size):
            for j in range(self.size):
                if (i - self.half_size) * tan_theta < abs(j - self.half_size):
                    im[i, j] = 0
        if im.sum() >= 2:
            index = np.where(im == 1)
            sample = np.c_[index[0], index[1]]
            kmeans.fit(sample)
            center_1 = np.around(kmeans.cluster_centers_[0]).astype(int)
            center_2 = np.around(kmeans.cluster_centers_[1]).astype(int)
            if show:
                # im = np.copy(img)
                im[center_1[0] - 3: center_1[0] + 3, center_1[1] - 3:center_1[1] + 3] = 1
                im[center_2[0] - 3:center_2[0] + 3, center_2[1] - 3:center_2[1] + 3] = 1
                im[self.half_size - 1:self.half_size + 1, self.half_size - 1:self.half_size + 1] = 1
                # im_show = im + img
                im_show = im
                size = int(self.size * self.scope)
                im_show = Image.fromarray(im_show)
                im_show = im_show.resize((size, size), Image.BILINEAR)
                im_show = np.array(im_show)
                cv2.imshow("leg", im_show)
                cv2.waitKey(1)
            if center_1[1] < center_2[1]:
                self.left_leg = self.center_point - center_1
                self.right_leg = self.center_point - center_2
            else:
                self.left_leg = self.center_point - center_2
                self.right_leg = self.center_point - center_1
            return center_1, center_2
        else:
            infinite_far = np.array([-180, -180])
            self.left_leg = infinite_far
            self.right_leg = infinite_far
            return infinite_far, infinite_far

    def scan_procedure(self,show: bool = False, is_record: bool = False, file_path:str=data_path):
        info = self.rplidar.get_info()
        print(info)
        health = self.rplidar.get_health()
        print(health)
        if is_record:
            data_path = file_path + os.path.sep + "leg.txt"
            file_leg = open(data_path, 'w')
        try:
            for i, scan in enumerate(self.rplidar.iter_scans(max_buf_meas=5000)):
                # print('%d: Got %d measurments' % (i, len(scan)))
                # print(scan)
                img = self.turn_to_img(scan)
                self.detect_leg(self.kmeans, img, show=show)
                # print(self.left_leg, self.right_leg)
                if is_record:
                    time_index = time.time()
                    leg_data = np.r_[self.left_leg, self.right_leg]
                    write_data = leg_data.tolist()
                    write_data.insert(0, time_index)
                    file_leg.write(str(write_data) + '\n')
                    file_leg.flush()

        except KeyboardInterrupt as e:
            self.rplidar.stop()
            self.rplidar.stop_motor()
            self.rplidar.disconnect()


if __name__ == "__main__":
    lidar = Leg_detector()
    lidar.scan_procedure(show=True)
