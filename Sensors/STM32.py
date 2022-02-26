import serial
# from Sensors import softskin, Infrared_Sensor
# from Driver import ControlOdometryDriver
from Sensors.SensorFunctions import *
from Sensors.SensorConfig import *
import Communication.State_client as csc
from global_variables import WalkerState

def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton

@singleton
class STM32Sensors():
    def __init__(self,serial_number: str = STM32_SERIAL_DESCRIPTION, baudrate: int = STM32_BAUDRATE):
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

    def UpdateDriver(self, linearVelocity: float, angularVelocity: float, distanceToCenter: int):
        if linearVelocity != self.vehicle_linear_velocity and angularVelocity != self.vehicle_angular_velocity and \
                distanceToCenter != self.vehicle_angular_distance:
            self.vehicle_linear_velocity = linearVelocity
            self.vehicle_angular_velocity = angularVelocity
            self.vehicle_angular_distance = distanceToCenter
            self.SetVehicleSpeed()

    def SetVehicleSpeed(self):
        self.serial.write(b's')
        buf = self.serial.read(1)
        count = 1
        for count in range(1, 100):
            if buf == b's':
                break
            else:
                count = count + 1
                buf = self.serial.read(1)
        para1 = str(self.vehicle_linear_velocity)
        para2 = str(self.vehicle_angular_velocity)
        para3 = str(self.vehicle_angular_distance)
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

    def STM_loop(self):
        try:
            while True:
                # TODO: change every data update
                self.GetSoftskinData()
                # self.skin.update_from_STM32(self.softskin_data)
                self.GetInfraredData()
                # self.Infrared.update_from_STM32(self.infrared_sensor_data)
                self.SetVehicleSpeed()
        except:
            print("STM32 Communication Error!")
            pass


if __name__ == "__main__":
    import threading
    import time
    STM32_instance = STM32Sensors()
    # STM32_thread = threading.Thread(target=STM32_instance.STM_loop, args=())
    # STM32_thread.start()
    for i in range(1):
        # linearVelocity: cm/s
        # angularVelocity: rad/s
        # distanceToCenter: cm      -:left +:right
        print(1)
        STM32_instance.UpdateDriver(linearVelocity=10,angularVelocity=0,distanceToCenter=0)
        time.sleep(2)
        # STM32_instance.UpdateDriver(linearVelocity=-10,angularVelocity=0,distanceToCenter=0)
        # time.sleep(2)
        STM32_instance.UpdateDriver(linearVelocity=0,angularVelocity=0,distanceToCenter=0)
        time.sleep(2)
        STM32_instance.UpdateDriver(linearVelocity=10,angularVelocity=0,distanceToCenter=0)
        time.sleep(2)
        STM32_instance.UpdateDriver(linearVelocity=0,angularVelocity=0,distanceToCenter=0)

