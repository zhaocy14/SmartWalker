import os,sys

#   DATA PATH
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
DATA_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."+ os.path.sep + "data")

#   Speaker USB port
SPEAKER_LOCATION_1 = "3.2-1"
SPEAKER_LOCATION_2 = "3.2-2"
SPEAKER_LOCATION_3 = "3.2-3"
SPEAKER_LOCATION_4 = "3.2-4"