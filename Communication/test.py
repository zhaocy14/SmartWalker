<<<<<<< HEAD

import os, sys
=======
import os, sys
import time
>>>>>>> f55e50a2dc3fef65693eb340ad1e57788e9c4f01
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

<<<<<<< HEAD
from Communication.Cpp_command import CppCommand
cco = CppCommand.get_instance()

if __name__ == "__main__":
    cco.start_navigation(mode="offline", testing="local", stdout=True)
=======
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
>>>>>>> f55e50a2dc3fef65693eb340ad1e57788e9c4f01
