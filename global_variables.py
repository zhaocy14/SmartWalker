import enum
from enum import Enum
from enum import auto
from enum import Flag

@enum.unique
class _Idle(Enum):
    NOT_CHARGING = 'not_charging'
    CHARGING = 'charging'
    
    
class _Running(Enum):
    VOICE_MENU = 'voice_menu'
    FRONT_FOLLOWING = 'front_following'
    DRAWING = 'drawing'
    NAVIGATING = 'navigating'
    BACK_HOME = 'back_home' # Going back to the power station
    MANUAL_MODE = 'manual_mode' # Going back to the power station
    
    
class _Error(Enum):
    UNKNOWN = 'unknown'
    
    
class WalkerState_old(Enum):
    IDLE = _Idle
    RUNNING = _Running
    ERROR = _Error
    UPDATING = 'updating'
    
    
class WalkerState(Flag):
    NOT_CHARGING = auto()
    CHARGING = auto()
    IDLE = NOT_CHARGING | CHARGING
    
    VOICE_MENU = auto()
    FRONT_FOLLOWING = auto()
    RUNNING = VOICE_MENU | FRONT_FOLLOWING
    
    ERROR = auto()
    
    UPDATING = auto()