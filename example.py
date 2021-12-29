#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
import Communication.State_client as csc
from global_variables import WalkerState

state_client = csc.StateClient.get_instance()
# print(state_client.change_walker_state(WalkerState.IDLE.CHARGING))
print(WalkerState.IDLE.CHARGING is WalkerState.IDLE.CHARGING)