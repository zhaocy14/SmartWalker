
import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Modules.Driver_recv import DriverRecv

if __name__ == "__main__":
    drv_recv = DriverRecv()
    drv_recv.start()