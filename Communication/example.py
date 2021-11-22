import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Cpp_command import CppCommand
from Communication.Receiver import Receiver
cco = CppCommand.get_instance(lidar_port="/dev/ttyUSB2", imu_port="/dev/ttyUSB1")

if __name__ == "__main__":
    # cco.start_navigation(mode="offline", testing="local", stdout=False, map_file="map.2021-09-25.070808")
    
    '''
        Start map drawing example
        **If wanted to use the latest map file, can use map_file="latest"
    '''
    # cco.start_navigation(stdout=False, driver_ctrl=False, ir_sensor=False, map_file="map.2021-09-25.070808")
    # recvObj = Receiver(mode="local")
    # # recvObj.start_DriverControl()
    # for _ in recvObj.start_Pose():
    #     print("testing", _)
    
    '''
        Start map drawing example
    '''
    # cco.start_drawing(stdout=True)
    
    '''
        Start sensors command example
    '''
    cco.start_sensors(stdout=True)
    # cco.start_navigation(stdout=True, driver_ctrl=False, ir_sensor=False, map_file="map.2021-09-25.070808")

    # cco.start_sensors(stdout=True)
    # import time
    # time.sleep(5)
    # cco.start_drawing(stdout=True)