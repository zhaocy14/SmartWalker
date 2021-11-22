import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Receiver import Receiver

recvObj = Receiver(mode="local")
for _ in recvObj.start_Pose():
        print("testing", _)