
import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Modules.Variables import *
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()

for topic, message in rzo.start(topic=pose_topic):
    print(message)