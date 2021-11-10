import os, sys
import time
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Modules.Variables import *
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()

current_time = time.time()
last_scan = 0
for scan in rzo.startLidar():
    if float(scan["theta"]) > last_scan:
        last_scan = float(scan["theta"])
    else:
        print(1/(time.time()-current_time))
        current_time = time.time()
        last_scan = 0


# for topic, message in rzo.start(topic=pose_topic):
#     print(message)