
import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Cpp_command import CppCommand
from Communication.Receiver import Receiver
cco = CppCommand.get_instance()

if __name__ == "__main__":
    # cco.start_navigation(mode="offline", testing="local", stdout=True)
    cco.start_navigation(stdout=False)