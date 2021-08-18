import serial
import os
import numpy as np
import time
import threading
import tensorflow as tf
from Sensors import IRCamera, softskin
from Network import FrontFollowingNetwork as FFL
from Driver import ControlOdometryDriver as CD
import cv2 as cv

if __name__ == "__main__":
  win_width = 10
  tf_model = FFL.FrontFollowing_Model(win_width=win_width)
  weight_path = "./Network/checkpoints/FrontFollowing"
  tf_model.model.load_weights(weight_path)

  skin = softskin.SoftSkin()
  IRCamera = IRCamera.IRCamera()
  skin.build_base_line_data()

  max_ir = 55
  min_ir = 10

  def binarization(img):
      """according to an average value of the image to decide the threshold"""
      if len(img.shape) == 2:
          threshold = max(img.mean() + 1.4, 23)
          img[img < threshold] = 0
          img[img >= threshold] = 1
      return img

  def filter(img):
      img_new = np.copy(img)
      img_new = img_new.reshape((24, 32))
      filter_kernel = np.ones((2, 2)) / 4
      """other filters"""
      # filter_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/10
      # filter_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
      for j in range(1):
          img_new = cv.filter2D(img_new, -1, filter_kernel)
      img_new = img_new.flatten()
      return img_new


  ir_threshold = 27
  ir_data_width = 768
  skin_data_width = 32
  buffer_length = win_width
  buffer = np.zeros((buffer_length * (ir_data_width + skin_data_width), 1))

  thread_skin = threading.Thread(target=skin.read_and_record,args=())
  thread_skin.start()
  cd = CD.ControlDriver()
  # thread_control_driver = threading.Thread(target=cd.control_part, args=())
  # thread_control_driver.start()


  while True:
      # present_time = time.time()
      IRCamera.get_irdata_once()
      if len(IRCamera.temperature) == 768:
          normalized_temperature = np.array(IRCamera.temperature).reshape((ir_data_width,1))
          idx = normalized_temperature < ir_threshold
          normalized_temperature[idx] = 0
          idx = normalized_temperature >= ir_threshold
          normalized_temperature[idx] = 1
          # normalized_temperature = (normalized_temperature-min_ir)/(max_ir-min_ir)
          buffer[0:(buffer_length-1)*ir_data_width,0] = buffer[ir_data_width:buffer_length*ir_data_width,0]
          buffer[(buffer_length-1)*ir_data_width:buffer_length*ir_data_width] = normalized_temperature


          """skin part start index"""
          SSI = buffer_length*ir_data_width

          buffer[SSI:SSI+(buffer_length-1)*skin_data_width,0] = \
              buffer[SSI+skin_data_width:SSI+buffer_length*skin_data_width,0]
          buffer[SSI+(buffer_length-1)*skin_data_width:SSI+buffer_length*skin_data_width] = \
              np.array(skin.temp_data).reshape((skin_data_width,1))
          # buffer.shape = (4000,1)
          buffer[SSI:SSI + buffer_length * skin_data_width, 0] = 0

          predict_buffer = buffer.reshape((-1,buffer_length*(ir_data_width+skin_data_width),1))
          result = tf_model.predict(predict_buffer)
          max_result = result.max()
          # print(max_result)
          if max_result == result[0, 0]:
              print("still!")
              cd.speed = 0
              cd.omega = 0
              cd.radius = 0
          elif max_result == result[0, 1]:
              print("forward!")
              cd.speed = 0.1
              cd.omega = 0
              cd.radius = 0
          elif max_result == result[0, 2]:
              print("turn left!")
              cd.speed = 0
              cd.omega = 0.1
              cd.radius = 2
          elif max_result == result[0, 3]:
              print("turn right!")
              cd.speed = 0
              cd.omega = -0.1
              cd.radius = 2
          elif max_result == result[0, 4]:
              print("yuandi left")
              cd.speed = 0
              cd.omega = 0.2
              cd.radius = 0
          elif max_result == result[0, 5]:
              print("yuandi right")
              cd.speed = 0
              cd.omega = -0.2
              cd.radius = 0
          elif max_result == result[0, 6]:
              print("backword")
              cd.speed = -0.1
              cd.omega = 0
              cd.radius = 0
          # print(1/(time.time()-present_time))
          # present_time = time.time()









