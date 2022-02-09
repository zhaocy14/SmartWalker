import serial
from Sensors import softskin, Infrared_Sensor
from Sensors.SensorFunctions import *
from Sensors.SensorConfig import *
from Driver import ControlOdometryDriver
import Communication.State_client as csc
from global_variables import WalkerState


class STM32_communication(object):

    def __init__(self,serial_number: str = STM32_SERIAL_NUM, baudrate: int = 115200):
        self.port_name, self.port_list = detect_serials(port_key=serial_number, sensor_name="STM32")
        self.serial = serial.Serial(port=self.port_name, baudrate=baudrate)
        # sensor
        # TODO: how to update data for these sensors
        # self.skin = skin
        # self.Infrared = Infrared
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
        self.state_client = csc.StateClient.get_instance()

    # def GetCameraData(self):
    #     # will not use anymore
    #     count = 1
    #     self.serial.write(b'c')
    #     buf = self.serial.read(1)
    #     for count in range(1, 100):
    #         if buf == b'c':
    #             break
    #         else:
    #             count = count + 1
    #             buf = self.serial.read(1)
    #     if count >= 100:
    #         print("Communication timeout, please retry")
    #     else:
    #         data = self.serial.read(3072)
    #         self.camera_data = data.hex()
    #         return data

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

    def UpdateVehicleSpeed(self, linearVelocity: float, angulrVelocity: float, distanceToCenter: int):
        self.vehicle_linear_velocity = linearVelocity
        self.vehicle_angular_velocity = angulrVelocity
        self.vehicle_angular_distance = distanceToCenter

    def SetVehicleSpeed(self, linearVelocity: float, angulrVelocity: float, distanceToCenter: float):
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

    def GetChargingState(self):
        # Todo: Get the charging state
        is_charging = False
        self.state_client.set_charging(is_charging)

    def GetPowerLevel(self):
        # Todo: Get the power level
        power_level = 75
        self.state_client.set_power_level(power_level)

    def main_communicate(self):
        try:
            while True:
                # TODO: change every data update
                # self.GetSoftskinData()
                # self.skin.update_from_STM32(self.softskin_data)
                # self.GetInfraredData()
                # self.Infrared.update_from_STM32(self.infrared_sensor_data)
                self.SetVehicleSpeed()
        except:
            print("STM32 Communication Error!")
            pass


if __name__ == "__main__":

    import threading
    import time

    STM32_instance = STM32_communication()
    thread_STM = threading.Thread(target=STM32_instance.main_communicate, args=())
    thread_STM.start()

    STM32_instance.SetVehicleSpeed(linearVelocity=0.1,angulrVelocity=0,distanceToCenter=0)
    time.sleep(3)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=0,distanceToCenter=0)
    time.sleep(1)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=0.3,distanceToCenter=0)
    time.sleep(3)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=0,distanceToCenter=0)
    time.sleep(1)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=-0.3,distanceToCenter=0)
    time.sleep(3)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=0,distanceToCenter=0)
    time.sleep(1)
    STM32_instance.SetVehicleSpeed(linearVelocity=0,angulrVelocity=0.2,distanceToCenter=50)
    time.sleep(3)
    STM32_instance.SetVehicleSpeed(linearVelocity=-0.4,angulrVelocity=0,distanceToCenter=0)
    time.sleep(2)
    STM32_instance.SetVehicleSpeed(linearVelocity=0, angulrVelocity=0, distanceToCenter=0)
