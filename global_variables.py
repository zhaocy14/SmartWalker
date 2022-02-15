#
# Created on Thu Dec 30 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import enum
from enum import Enum, EnumMeta, auto, Flag


class MyEnumMeta(enum.EnumMeta): 
    def __contains__(cls, item): 
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


@enum.unique
class _IDLE(Enum, metaclass=MyEnumMeta):
    DEFAULT = 'idle'
    NOT_CHARGING = 'not_charging'
    CHARGING = 'charging'
    
    
@enum.unique    
class _RUNNING(Enum, metaclass=MyEnumMeta):
    VOICE_MENU = 'voice_menu'
    FRONT_FOLLOWING = 'front_following'
    DRAWING = 'drawing'
    NAVIGATING = 'navigating'
    BACK_HOME = 'back_home' # Going back to the power station
    MANUAL_MODE = 'manual_mode' # Going back to the power station
    

@enum.unique    
class _ERROR(Enum, metaclass=MyEnumMeta):
    UNKNOWN = 'unknown'
    
    
@enum.unique    
class _UPDATE(Enum, metaclass=MyEnumMeta):
    UPDATING = 'updating'
    
 
class WalkerState():
    IDLE = _IDLE
    RUNNING = _RUNNING
    ERROR = _ERROR
    UPDATE = _UPDATE
    
    
    def __init__(self) -> None:
        pass
    
    
    @staticmethod
    def getByValue(str=""):
        if str in _IDLE._value2member_map_:
            return _IDLE(str)
        elif str in _RUNNING._value2member_map_:
            return _RUNNING(str)
        elif str in _ERROR._value2member_map_:
            return _ERROR(str)
        elif str in _UPDATE._value2member_map_:
            return _UPDATE(str)
        else:
            return False


@enum.unique
class WalkerPort(Enum):
    RECV = '5454'
    TRANSMIT = '5455'
    LIDAR = '5450'
    IMU = '5451' 
    WALKER_STATE = '5456'
    
# @enum.unique    
# class WalkerState(Enum):
#     NOT_CHARGING = 'not_charging'
#     CHARGING = 'charging'
#     IDLE = (NOT_CHARGING, CHARGING)
    
    # VOICE_MENU = auto()
    # FRONT_FOLLOWING = auto()
    # RUNNING = VOICE_MENU | FRONT_FOLLOWING
    
    # ERROR = auto()
    
    # UPDATING = auto()