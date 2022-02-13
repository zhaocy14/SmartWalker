import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(os.path.dirname(pwd)) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(father_path + os.path.sep + ".." +
                    os.path.sep + "data"))

import numpy as np
import math
import cv2
import threading
from PIL import Image
from Sensors import LiDAR
# import rplidar
import time
from sklearn.cluster import KMeans
from Communication.Modules.Receive import ReceiveZMQ


class Leg_detector(object):
    def __init__(self, is_show: bool = False, is_zmq: bool = False):
        if not is_zmq:
            self.rplidar = LiDAR.lidar()
        else:
            self.rplidar = []

        self.scan_raw_data = np.zeros((1, 1))
        self.kmeans = KMeans(n_clusters=2)
        self.left_leg = np.zeros((2))
        self.right_leg = np.zeros((2))
        self.scope = 2

        # unit cm
        # for 2D scan images sizes
        self.size = 300
        self.half_size = int(self.size / 2)
        # old version of filtering useless data
        self.column_boundry = self.half_size - 20
        self.filter_theta = 150
        self.bottom_boundary = self.half_size - 100
        # new version of filtering useless data
        self.walker_top_boundary = 12
        self.walker_bottom_boundary = 50
        self.walker_left_boundary = 34
        self.walker_right_boundary = 38
        # cecnter point is the geometry center of the walker
        self.center_point = np.array([self.half_size + 45, self.half_size])
        self.is_show = is_show
        # zmq part
        self.rzo = ReceiveZMQ.get_instance()
        self.zmq_temp_list = []
        self.zmq_scan_list = []
        self.theta_flag = 0
        # obstacle part
        self.obstacle_array = np.zeros((1, 5))  # 0 means no obstacle, else means yes
        self.obstacle_distance = 15  # 15 cm detection

    def turn_to_img(self, original_list: list, show: bool = False):
        """turn the scan data list into a ndarray"""
        img = np.zeros((self.size, self.size))
        for i in range(len(original_list)):
            theta = original_list[i][1]
            theta = -theta / 180 * math.pi
            distance = original_list[i][2] / 10  # unit: mm->cm
            # distance = original_list[i][2]  # unit: mm
            # turn distance*theta -> x-y axis in the scan image
            index_x = int(distance * math.cos(theta) + self.half_size)
            index_y = int(distance * math.sin(theta) + self.half_size)
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

    def detect_leg_boundary_version(self, kmeans: KMeans, img: np.ndarray, theta: float = 160, show: bool = False):
        theta = theta / 180 * math.pi
        tan_theta = math.tan(theta / 2)
        leg_img = np.zeros((self.walker_top_boundary + self.walker_bottom_boundary + 50,
                            self.walker_left_boundary + self.walker_right_boundary))
        leg_img[:, :] = img[self.half_size - self.walker_top_boundary:self.half_size + self.bottom_boundary + 50,
                        self.half_size - self.walker_left_boundary:self.half_size + self.walker_right_boundary]
        for i in range(self.walker_top_boundary, self.walker_bottom_boundary + 50):
            for j in range(self.walker_left_boundary + self.walker_right_boundary):
                if i * tan_theta < abs(j - self.walker_left_boundary):
                    leg_img[i, j] = 0
        detect_leg_img = np.zeros((leg_img.shape))
        detect_leg_img[18:-1, 15:-15] = leg_img[18:-1, 15:-15]
        # print(leg_img.max(),detect_leg_img.max())
        if detect_leg_img.sum() >= 2:
            index = np.where(detect_leg_img == 1)
            sample = np.c_[index[0], index[1]]
            kmeans.fit(sample)
            center_1 = np.around(kmeans.cluster_centers_[0]).astype(int)
            center_2 = np.around(kmeans.cluster_centers_[1]).astype(int)
            if show:
                # to show the leg position in the image
                leg_img[center_1[0] - 1: center_1[0] + 1, center_1[1] - 1:center_1[1] + 1] = 1
                leg_img[center_2[0] - 1:center_2[0] + 1, center_2[1] - 1:center_2[1] + 1] = 1
                # to show the LiDAR point in the image
                leg_img[self.walker_top_boundary - 1:self.walker_top_boundary + 1,
                self.walker_left_boundary - 1:self.walker_left_boundary + 1] = 1
                # im_show = im + img
                im_show = leg_img
                # transform to Image to change the size of the print image
                im_show = Image.fromarray(im_show)
                img_scope = 5
                img_size_row = (self.walker_top_boundary + self.walker_bottom_boundary) * img_scope
                img_size_column = (self.walker_left_boundary + self.walker_right_boundary) * img_scope
                im_show = im_show.resize((img_size_row, img_size_column), Image.BILINEAR)
                im_show = np.array(im_show)
                cv2.imshow("leg", im_show)
                cv2.waitKey(1)
            self.center_point[0] = self.walker_top_boundary + self.walker_bottom_boundary
            self.center_point[1] = self.walker_left_boundary
            if center_1[1] < center_2[1]:
                self.left_leg = self.center_point - center_1
                self.right_leg = self.center_point - center_2
            else:
                self.left_leg = self.center_point - center_2
                self.right_leg = self.center_point - center_1
            return self.left_leg, self.right_leg
        else:
            infinite_far = np.array([-180, -180])
            self.left_leg = infinite_far
            self.right_leg = infinite_far
            return infinite_far, infinite_far

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

    def detect_obstacle(self, img: np.ndarray):
        front_od = self.obstacle_distance + 10
        side_od = self.obstacle_distance - 2
        obstacle_area = img[self.half_size - self.walker_top_boundary - front_od:
                            self.half_size + self.bottom_boundary,
                        self.half_size - self.walker_left_boundary - side_od:
                        self.half_size + self.walker_right_boundary + side_od]
        self.obstacle_array[0, 0] = obstacle_area[0:front_od, 0:side_od].sum()
        self.obstacle_array[0, 1] = obstacle_area[0:front_od - 3, side_od:-side_od].sum()
        self.obstacle_array[0, 2] = obstacle_area[0:front_od, -side_od:-1].sum()
        self.obstacle_array[0, 3] = obstacle_area[front_od:-1, 0:side_od].sum()
        self.obstacle_array[0, 4] = obstacle_area[front_od:-1, -side_od:-1].sum()
        # print(self.obstacle_array)

    def scan_procedure(self, show: bool = False, is_record: bool = False, file_path: str = data_path):
        while True:
            try:
                info = self.rplidar.get_info()
                print(info)
                health = self.rplidar.get_health()
                print(health)
                if is_record:
                    data_path = file_path + os.path.sep + "leg.txt"
                    file_leg = open(data_path, 'w')
                for i, scan in enumerate(self.rplidar.iter_scans(max_buf_meas=500)):
                    self.scan_raw_data = np.array(scan)
                    img = self.turn_to_img(scan)
                    self.detect_obstacle(img)
                    self.detect_leg(self.kmeans, img, show=show)
                    if is_record:
                        time_index = time.time()
                        leg_data = np.r_[self.left_leg, self.right_leg]
                        write_data = leg_data.tolist()
                        write_data.insert(0, time_index)
                        file_leg.write(str(write_data) + '\n')
                        file_leg.flush()
                    # self.rplidar.clean_input()

            except BaseException as be:
                # pass
                self.rplidar.clean_input()
                self.rplidar.stop()
                # if stop motor, this will
                # self.rplidar.stop_motor()
                # self.rplidar.disconnect()

    def zmq_get_one_round(self, zmq_data: dict):
        """use zmq to get lidar data from C++"""
        theta = float(zmq_data["theta"])
        dist = float(zmq_data["dist"])
        quality = float(zmq_data["q"])
        if theta < self.theta_flag:
            self.zmq_scan_list = self.zmq_temp_list
            self.zmq_temp_list = []
        self.theta_flag = theta
        self.zmq_temp_list.append([quality, theta, dist])

    def zmq_scan(self, show: bool = False, is_record: bool = False, file_path: str = data_path):
        """main procedure of scanning legs and the environment"""
        if is_record:
            data_path = file_path + os.path.sep + "leg.txt"
            file_leg = open(data_path, 'w')
        while True:
            try:
                for scan in self.rzo.startLidar():
                    self.zmq_get_one_round(scan)
                    if len(self.zmq_temp_list) == 1:
                        self.scan_raw_data = np.array(self.zmq_scan_list)
                        img = self.turn_to_img(self.zmq_scan_list)
                        self.detect_leg_boundary_version(self.kmeans, img, show=show)
                        if is_record:
                            time_index = time.time()
                            leg_data = np.r_[self.left_leg, self.right_leg]
                            write_data = leg_data.tolist()
                            write_data.insert(0, time_index)
                            file_leg.write(str(write_data) + '\n')
                            file_leg.flush()
            except BaseException as be:
                pass


if __name__ == "__main__":
    # LD = Leg_detector(is_zmq=True)
    # LD.zmq_scan(show=True)
    LD = Leg_detector(is_zmq=False)
    LD.scan_procedure(show=True, is_record=False)
