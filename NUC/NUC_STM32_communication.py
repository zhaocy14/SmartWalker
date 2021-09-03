import serial
import time
import serial.tools.list_ports


ser = serial.Serial()
ser_tol = serial.tools.list_ports

ports = ser_tol.comports()
for port, desc, hwid in ports:
    print(port)
ser.port = port
ser.baudrate = 921600
ser.timeout = 100
ser.open()

def SetVehicleSpeed(speed:float=0.0, omega:float=0.0, distance:int=0):
    para1 = str(speed)
    para2 = str(omega)
    para3 = str(distance)
    send_data = 's' + para1 + ' ' + para2 + ' ' + para3 + '\n'
    ser.write(bytes(send_data,'UTF-8'))
    print(send_data)

def Getdata():
    buffer = ser.read(1)
    print(buffer)
    while buffer != b's':
        buffer = ser.read(1)
        print(buffer)
    data = ser.readline()
    return data

if __name__ == "__main__":
    for i in range(6):
        time.sleep(1)
        speed = -10
        omega = 0
        distance = 0
        SetVehicleSpeed(speed,omega,distance)
        print(Getdata())
    SetVehicleSpeed(0, 0, 0)

