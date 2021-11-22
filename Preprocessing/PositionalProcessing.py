import cv2 as cv
import numpy as np
import math
from typing import List, Tuple, NoReturn


class CC_matrix(object):

    def __init__(self):
        self.Mint = np.zeros((3,4))
        self.Mext = np.zeros((4,4))
        self.theta_x = 145
        self.theta_y = 0
        self.theta_z = 0
        self.xc = 0
        self.yc = -0.32
        self.zc = 0.30
        self.fx = 11.45
        self.fy = 13.03
        self.ox = 16
        self.oy = 13.13
        self.get_Mint(fx=self.fx,fy=self.fy,ox=self.ox,oy=self.oy)
        self.get_Mext(theta_x=self.theta_x,theta_y=self.theta_y,theta_z=self.theta_z,
                      xc=self.xc,yc=self.yc,zc=self.zc)
        return

    def get_Mext(self, theta_x,theta_y,theta_z,xc,yc,zc):
        theta_x = theta_x/180*math.pi
        theta_y = theta_y/180*math.pi
        # theta_z = theta_z/180*math.pi
        R_x = [[1, 0, 0],
               [0,math.cos(theta_x),math.sin(theta_x)],
               [0,-math.sin(theta_x),math.cos(theta_x)]]
        R_y = [[math.cos(theta_y),0,-math.sin(theta_y)],
               [0,1,0],
               [math.sin(theta_y),0,math.cos(theta_y)]]
        R_z = [[math.cos(theta_z),math.sin(theta_z),0],
               [-math.sin(theta_z),math.cos(theta_z),0],
               [0,0,1]]
        R_x = np.array(R_x)
        R_y = np.array(R_y)
        R_z = np.array(R_z)
        Rotation = np.matmul(np.matmul(R_x, R_y), R_z)
        tc = np.array([[-xc],[-yc],[-zc]])
        tc = np.matmul(Rotation,tc)
        self.Mext[0:3,0:3] = Rotation
        self.Mext[0:3,3] = tc.reshape((3,))
        self.Mext[3,3] = 1

    def get_Mint(self,fx,fy,ox,oy):
        self.Mint[0,0] = fx
        self.Mint[0,2] = ox
        self.Mint[1,1] = fy
        self.Mint[1,2] = oy
        self.Mint[2,2] = 1

    def get_delta_uv(self,dx,dy,dz,dtheta):
        list_a = [float(dx),float(dy),float(dz),float(1)]
        print(list_a)
        dlocation = np.array(list_a)
        self.get_Mext(self.theta_x,self.theta_y,dtheta,self.xc,self.yc,self.zc)
        duv = np.matmul(self.Mext,dlocation)
        duv = np.matmul(self.Mint,duv)
        duv = np.round(duv)
        print(duv)
        return duv

    def get_img_one_flame(self,img,dx,dy,dz,dtheta):
        duv = self.get_delta_uv(dx=dx,dy=dy,dz=dz,dtheta=dtheta)
        new_img = np.zeros((img.shape[0],img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img.shape[0] > i + duv[0] >= 0 and img.shape[1] > j + duv[1] >= 0:
                        new_img[i,j] = img[int(i+duv[0]),int(j+duv[1])]
        return new_img

    def get_dxyz_series(self,position_list,buffer_length):
        new_postition_list = list.copy(position_list)
        for i in range(buffer_length):
            j = i
            while j < buffer_length:
                new_postition_list[i] += position_list[j]
                j += 1


    def get_img_multi_flame(self,img_list,position_list,buffer_length):
        new_img_list = []
        for i in range(buffer_length):
            [dx,dy,dz] = position_list[i]
            img = img_list[i]
            new_img_list.append(self.get_img_one_flame(img,dx,dy,dz))
        factor = []
        for i in range(buffer_length):
            factor.append((i+1)/sum(range(buffer_length+1)))
        fixed_img = np.zeros((img_list[0].shape[0],img_list[0].shape[1]))
        for i in range(buffer_length):
            fixed_img = fixed_img + factor[i]*new_img_list[i]
        return fixed_img

    def calculate_the_world_coordinate(self,U,V,height_of_camera=30, angle_of_camera=35):
        """calculate the ground plane formula in the camera coordinate"""
        angle_of_camera = angle_of_camera/180*math.pi
        """calculate the interaction point"""
        x0 = 0
        y0 = -height_of_camera/math.sin(angle_of_camera)
        z0 = 0
        """calculate the formal vector of the ground plane"""
        A = 0
        B = height_of_camera * math.sin(angle_of_camera)
        C = height_of_camera * math.cos(angle_of_camera)

        v = (U - self.ox)/self.fx
        u = (V - self.oy)/self.fy

        """establish fomular to solve the X Y  coordinate"""
        k0 = A/C*x0+B/C*y0+z0
        A1 = u/C*A+1
        B1 = u/C*B
        C1 = k0*u
        A2 = v/C*A
        B2 = v/C*B+1
        C2 = k0*v
        """calculate the position of the feet in the camera coordinate"""
        x = (B1*C2-B2*C1)/(A2*B1-B2*A1)
        y = (C2*A1-A2*C1)/(A1*B2-A2*B1)
        z = -(A*(x-x0)+B*(y-y0))/C + z0
        """calculate the position of the feet in the world coordinate"""
        camera_coordinate = np.array([x,y,z,1]).reshape((4,1))
        world_coordinate = np.matmul(np.linalg.inv(self.Mext),camera_coordinate)

        """return a [x,y,z] in the world coordinate"""
        return world_coordinate[0:3]


class User_Postition_Estimate(object):
    def __init__(self, img: np.ndarray = np.zeros((24,32))):
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
            threshold = max(img.mean() + self.bias_for_threshold,23)
            img[img < threshold] = 0
            img[img >= threshold] = 1
            img = img.reshape((img.shape[0],img.shape[1]))
            """transform the format of the element of the raw data"""
            img = (img*255).astype(np.uint8)

        elif len(img.shape) == 3:
            """suppose read from a jpg file with three channels"""
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, img = cv.threshold(img, 127, 255, 0)
            print(type(img[0][0]))
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

    def get_COM(self, show: bool = True) -> NoReturn:
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
        contour_dict = dict(zip(list_area, contours))
        contour_dict = sorted(contour_dict.items(), key=lambda k: k[0], reverse=True)
        contours = []
        if len(contour_dict) == 0:
            self.user_x, self.user_y = 0, 0
        elif len(contour_dict) == 1:
            contours.append(contour_dict[0][1])
        else:
            if abs(contour_dict[0][0] - contour_dict[1][0]) <= 100:

                contours.append(contour_dict[0][1])
                contours.append(contour_dict[1][1])
            else:
                contours.append(contour_dict[0][1])

        COM = []
        if len(contours) != 0:
            for i in contours:
                contours_mom = cv.moments(i)
                COM_i = (
                int(contours_mom['m10'] / contours_mom['m00']), int(contours_mom['m01'] / contours_mom['m00']))
                COM.append(COM_i)

        if len(COM) == 1:
            self.user_x = COM[0][0]
            self.user_y = COM[0][1]
        elif len(COM) == 2:
            self.user_x = (COM[0][0] + COM[1][0]) / 2
            self.user_y = (COM[0][1] + COM[1][1]) / 2

        if show:
            show_img = np.copy(self.image)
            show_img = cv.drawContours(show_img, contours, -1, (0, 0, 255), 4)
            for i in COM:
                cv.circle(show_img, i, 2, (0, 0, 255), 2)
            cv.imshow('drawimg', show_img)
            cv.waitKey(0)

if __name__ == '__main__':
    tr = CC_matrix()
    # print(tr.Mext,'\n')
    # print(np.linalg.inv(tr.Mext),'\n')
    # print(np.transpose(tr.Mext),'\n')
    # print(np.matmul(tr.Mext,np.linalg.inv(tr.Mext)),'\n')
    # print(np.matmul(tr.Mext, np.transpose(tr.Mext)), '\n')
    #
    # print(np.matmul(tr.Mext[0:3,0:3],np.transpose(tr.Mext[0:3,0:3])))
    def get_data(direction):
        # 原始信息获取
        file = open(direction)
        list_ir_data = file.readlines()
        lists = []
        for lines in list_ir_data:
            lines = lines.strip("\n")
            lines = lines.strip('[')
            lines = lines.strip(']')
            lines = lines.split(", ")
            lists.append(lines)
        file.close()
        array_data = np.array(lists)
        rows_data = array_data.shape[0]
        columns_data = array_data.shape[1]
        data = np.zeros((rows_data, columns_data))
        for i in range(rows_data):
            for j in range(columns_data):
                data[i][j] = float(array_data[i][j])
        return data


    """load the data"""
    direction_ir_data = "./Record_data/data/ir_data.txt"
    ir_data = get_data(direction_ir_data)
    img = np.zeros((24, 32))
    UP = User_Postition_Estimate(img)
    for i in range(ir_data.shape[0]):
        img = ir_data[i][1:ir_data.shape[1]]
        img = img.reshape((24,32))
        UP.get_new_img(img)
        UP.get_COM(False)

        print(tr.calculate_the_world_coordinate(UP.user_x,UP.user_y))

