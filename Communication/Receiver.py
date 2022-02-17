import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Modules.Variables import *
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()
# rzo = ReceiveZMQ.get_instance(port="5456")
from Communication.Modules.Driver_recv import DriverRecv
from Communication.Modules.Pose_recv import PoseRecv


class Receiver(object):
    def __init__(self, mode="online"):
        self.drv_recv = DriverRecv(mode=mode)
        self.pose_recv = PoseRecv()

    
    def start_DriverControl(self):
        self.drv_recv.start(use_thread=True)
    

    def start_Pose(self):
        # t1 = threading.Thread(target=self.pose_recv.start)
        # t1.start()
        return self.pose_recv.start()
    
    # def start_Lidar(self):
    #     for _ in rzo.startLidar():
    #         print("lidar data", _)


if __name__ == "__main__":
    recvObj = Receiver(mode="local")
