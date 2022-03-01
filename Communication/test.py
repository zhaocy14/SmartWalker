import os, sys
import time

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from threading import Thread
from multiprocessing import Process
from Communication.Modules.Receive import ReceiveZMQ

from global_variables import WalkerPort, CommTopic
rzo = ReceiveZMQ.get_instance()
# for topic, message in rzo.start_old(topic=CommTopic.POSE.value):
for topic, message in rzo.start(topics=[CommTopic.POSE.value]):
    print('-'*20,topic,'-'*20,message)