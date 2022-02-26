import serial
import serial.tools.list_ports
import os
import sys
import numpy as np
import time

def print_serial(port):
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


def detect_serials(description="target device", vid=0x10c4, pid=0xea60):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print_serial(port)

        if port.description.__contains__(description):
            port_path = port.device
            return port_path
        else:
            print("Cannot find the target device: %s" % description)
    return None


class HeartRate:
    def __init__(self,description="USB-SERIAL CH340"):
        self.port_name = detect_serials(description=description)
        self.baudrate = 9600
        self.serial = serial.Serial(port=self.port_name,baudrate=self.baudrate)
        self.period = 0.25 #communication period is limited to 0.3s
        self.HR = 0 #heart rate
        self.SPO2 = 0 #blood oxygen
        self.check_status()
        self.reset()
        self.set_HR_alert(alert_up=100,alert_down=50)
        self.set_SPO2_alert(alert_up=110, alert_down=85)

    def check_status(self,show=True):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('AT\r\n'.encode())
        status = self.serial.readline().decode("utf-8").strip('\r\n')
        while True:
            time.sleep(0.2)
            if status == 'OK':
                if show:
                    print("Status:OK")
                break
            else:
                print("Status:Error. Please check the sensor")
        self.serial.write('AT+VERSION\r\n'.encode())
        if show:
            print("Version:"+self.serial.readline().decode("utf-8"))

    def reset(self, show = True):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('AT+RESET\r\n'.encode())
        result = self.serial.readline().decode("utf-8").strip('\r\n')
        if result == 'OK':
            if show:
                print("Reset done.")

    def get_HeartRate(self):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('AT+HEART\r\n'.encode())
        self.HR = self.serial.readline().decode("utf-8")
        self.HR = self.HR.strip('\r\n').strip('+HEART=')
        if self.HR != 'NULL':
            self.HR = int(self.HR)
        else:
            self.HR = 0

    def get_SPO2(self):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('AT+SPO2\r\n'.encode())
        self.SPO2 = self.serial.readline().decode("utf-8")
        self.SPO2 = self.SPO2.strip('\r\n').strip('+SPO2=')
        if self.SPO2 != 'NULL':
            self.SPO2 = int(self.SPO2)
        else:
            self.SPO2 = 0

    def set_HR_alert(self, alert_up=100, alert_down=50,show=False):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write(('AT+HERRUP+'+str(alert_up)+'\r\n').encode())
        self.serial.readline()
        self.serial.write(('AT+HERRDOWN+' + str(alert_down) + '\r\n').encode())
        self.serial.readline()
        if show:
            print("Heart rate alert set.")

    def set_SPO2_alert(self, alert_up=110, alert_down=88, show=False):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write(('AT+SERRUP+'+str(alert_up)+'\r\n').encode())
        self.serial.readline()
        self.serial.write(('AT+SERRDOWN+' + str(alert_down) + '\r\n').encode())
        self.serial.readline()
        if show:
            print("SPO2 alert set.")


    def get_TEMPRETURE(self):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('AT+T\r\n'.encode())
        print("Tempreture:"+self.serial.readline().decode("utf-8")+"℃")

if __name__ == '__main__':
    heartrate = HeartRate()
    while True:
        time.sleep(heartrate.period)
        heartrate.get_HeartRate()
        time.sleep(0.1)
        heartrate.get_SPO2()
        print("Heartrate:"+str(heartrate.HR)+"\tSPO2:"+str(heartrate.SPO2))

