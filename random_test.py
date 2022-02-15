import serial
from Sensors.SensorFunctions import *

def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class serial_test(object):

    def __init__(self):
        self.a = 1
        port,_=detect_serials(port_key="USB", sensor_name="22")
        self.ser = serial.Serial(port=port,baudrate=9600)


a = serial_test()
b = serial_test()

print(id(a),id(b))
