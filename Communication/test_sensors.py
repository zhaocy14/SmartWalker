
import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Cpp_command import CppCommand
cco = CppCommand.get_instance(lidar_port="/dev/ttyUSB0", imu_port="/dev/ttyUSB2")

if __name__ == "__main__":
    cco.start_sensors(stdout=True)