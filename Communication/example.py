import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import time

from Communication.Cpp_command import CppCommand
from Communication.Receiver import Receiver

if __name__ == "__main__":
    with CppCommand.get_instance() as cco:
        # cco.start_navigation(mode="offline", testing="local", stdout=False, map_file="map.2021-09-25.070808")
        '''
            Start navigation example
            **If wanted to use the latest map file, can use map_file="latest"
            **If needed to move the walker, set driver_ctrl=True
            **If needed the ir sensor, set ir_sensor=True
        '''
        cco.start_navigation(stdout=False, driver_ctrl=False, ir_sensor=False, map_file="map.2021-09-25.070808")
        time.sleep(5)
        # recvObj = Receiver(mode="local")
        # recvObj.start_DriverControl()
        # for _ in recvObj.start_Pose():
        #     print("testing", _)
        
        '''
            Start map drawing example
        '''
        # cco.start_drawing(stdout=True)
        
        '''
            Start sensors command example
        '''
        # cco.start_sensors(stdout=True)