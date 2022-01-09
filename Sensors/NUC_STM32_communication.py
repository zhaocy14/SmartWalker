import serial
import serial.tools.list_ports
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")

from Sensors import softskin, Infrared_Sensor
from Driver import ControlOdometryDriver

class STM32_communication(object):

    def __init__(self, skin:softskin.SoftSkin, Infrared:Infrared_Sensor.Infrared_Sensor,
                 Driver:ControlOdometryDriver.ControlDriver,serial_number:str='0669', baudrate:int=115200):

        self.port_name,self.port_list = self.detect_serials(number=serial_number)
        self.serial = serial.Serial(port=self.port_name,baudrate=baudrate)
        # sensor
        self.skin = skin
        self.Infrared = Infrared
        # data
        self.softskin_data = []
        self.infrared_sensor_data = []
        # Driver parameter
        self.vehicle_linear_velocity = 0.0
        self.vehicle_angular_velocity = 0.0
        self.vehicle_angular_distance = 0.0
        self.vehicle_on_off = False
        # camera no use
        self.camera_data = []

    """Print the port information"""
    def print_serial(self, port):
        print("---------------[ %s ]---------------" % port.name)
        print("Path: %s" % port.device)
        print("Descript: %s" % port.description)
        print("HWID: %s" % port.hwid)
        if not None == port.manufacturer:
            print("Manufacture: %s" % port.manufacturer)
        if not None == port.product:
            print("Product: %s" % port.product)
        if not None == port.interface:
            print("Interface: %s" % port.interface)
        print()

    """list all the port"""
    def detect_serials(self, number, vid=0x10c4, pid=0xea60):
        ports = serial.tools.list_ports.comports()
        port_cnt = 0
        port_list = []
        for port in ports:
            self.print_serial(port)
            if port.serial_number.__contains__(number):
                port_list = port.description
                port_path = port.device
                return port_path, port_list
            else:
                print("Cannot find the device: IR Camera")

    def GetCameraData(self):
        count = 1
        self.serial.write(b'c')
        buf = self.serial.read(1)
        for count in range(1, 100):
            if buf == b'c':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        if count >= 100:
            print("Communication timeout, please retry")
        else:
            data = self.serial.read(3072)
            self.camera_data = data.hex()
            return data

    def GetSoftskinData(self):
        count = 1
        self.serial.write(b'p')
        buf = self.serial.read(1)
        for count in range(1, 100):
            if buf == b'p':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        if count >= 100:
            print("Communication timeout, please retry")
        else:
            data = self.serial.read(128)
            self.softskin_data = data.hex()
            return data

    def GetInfraredData(self):
        count = 1
        self.serial.write(b'i')
        buf = self.serial.read(1)
        for count in range(1, 100):
            if buf == b'i':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        if count >= 100:
            print("Communication timeout, please retry")
        else:
            data = self.serial.read(32)
            self.infrared_sensor_data = data.hex()
            return data

    def UpdateVehicleSpeed(self, linearVelocity:float, angulrVelocity:float, distanceToCenter:int):
        self.vehicle_linear_velocity = linearVelocity
        self.vehicle_angular_velocity = angulrVelocity
        self.vehicle_angular_distance = distanceToCenter

    def SetVehicleSpeed(self,linearVelocity:float, angulrVelocity:float, distanceToCenter:int):
        self.serial.write(b's')
        buf = self.serial.read(1)
        count = 1
        for count in range(1, 100):
            if buf == b's':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        para1 = str(linearVelocity)
        para2 = str(angulrVelocity)
        para3 = str(distanceToCenter)
        dataToSend = para1 + ' ' + para2 + ' ' + para3 + '\n'
        if count >= 100:
            print("Coummunication timeout, please retry")
        else:
            self.serial.write(bytes(dataToSend, 'UTF-8'))
            data = self.serial.readline()
            return data

    def GetPressureData(self):
        count = 1
        self.serial.write(b'i')
        buf = self.serial.read(1)
        for count in range(1, 100):
            if buf == b'i':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        if count >= 100:
            print("Communication timeout, please retry")
        else:
            data = self.serial.read(32)
            self.infrared_sensor_data = data.hex()
            return data

    def main_communicate(self,):
        try:
            while True:
                self.GetSoftskinData()
                self.skin.update_from_STM32(self.softskin_data)
                self.GetInfraredData()
                self.Infrared.update_from_STM32(self.infrared_sensor_data)
                self.SetVehicleSpeed()
        except:
            print("STM32 Communication Error!")
            pass

if __name__ == "__main":
    STM32 = STM32_communication()