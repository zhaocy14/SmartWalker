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


class GPS:
    def __init__(self,description="USB-SERIAL CH340"):
        self.port_name = detect_serials(description=description)
        self.baudrate = 9600
        self.serial = serial.Serial(port=self.port_name,baudrate=self.baudrate)
        self.period = 0.25 #communication period is limited to 0.3s

        """message, please check the NEMA protocal file to see the definition"""
        self.GNGGA = ""
        self.GNGLL = ""
        self.GPGSA = ""
        self.BDGSA = ""
        self.GPGSV = ""
        self.BDGSV = ""
        self.GNRMC = ""
        self.GNVTG = ""
        self.GNZDA = ""
        self.GPTXT = ""

    def read_data(self):
        self.GNGGA = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GNGLL = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GPGSA = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.BDGSA = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GPGSV = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.BDGSV = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GNRMC = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GNVTG = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GNZDA = self.serial.readline().decode("utf-8").strip("\n").strip("$")
        self.GPTXT = self.serial.readline().decode("utf-8").strip("\n").strip("$")

    def print_all(self):
        print("\n\nALL data is:")
        print(self.GNGGA)
        print(self.GNGLL)
        print(self.GPGSA)
        print(self.BDGSA)
        print(self.GPGSV)
        print(self.GPGSV)
        print(self.GNRMC)
        print(self.GNVTG)
        print(self.GNZDA)
        print(self.GPTXT)



if __name__ == '__main__':
    my_gps = GPS()
    while True:
        my_gps.read_data()
        my_gps.print_all()

