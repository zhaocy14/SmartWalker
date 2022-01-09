import collections
import numpy as np
import statistics
import tensorflow as tf
from typing import List, Tuple, NoReturn
import time
import threading
# import softskin
from Sensors import IRCamera, softskin
from Network.FrontFollowingNetwork import FrontFollowing_Model as FFL
import PositionalProcessing as PP
from Driver.ControlOdometryDriver import ControlDriver


"""initialize camera"""
Camera = IRCamera.IRCamera()
skin = softskin.SoftSkin()
skin.build_base_line_data()
CD = ControlDriver()
UserPosition = PP.User_Postition_Estimate()
"""initialize the data register to store the data sequence"""
win_width = 10
ir_data_width = 768
skin_data_width = 32
buffer = np.zeros((win_width * (ir_data_width + skin_data_width), 1))
register_length = 3  # for RL learning state information storage
"""initialize the model"""
num_actions = 6
num_hidden_units = 128
FFL_model = FFL(win_width=win_width, is_multiple_output=True)
FFL_model.model.load_weights("./Network/checkpoints/FrontFollowing")
"""softskin data reading thread and Control driver thread"""
thread_skin = threading.Thread(target=skin.read_and_record, args=())
thread_skin.start()
thread_control_driver = threading.Thread(target=CD.control_part, args=())
thread_control_driver.start()
"""initialize the parameter of RL"""
min_episodes_criterion = 10000
max_episode = 10000
max_steps_per_episode = 20
reward_threshold = 195
running_reward = 0
gamma = 0.99

"""walker driver function"""

def walker_driver(action, CD: ControlDriver):
    if action == 0:
        print("still!")
        CD.speed = 0.0
        CD.omega = 0.0
        CD.radius = 0.0
    elif action == 1:
        print("forward!")
        CD.speed = 0.1
        CD.omega = 0.0
        CD.radius = 0.0
    elif action == 2:
        print("turn left!")
        CD.speed = 0.0
        CD.omega = 0.1
        CD.radius = 2.0
    elif action == 3:
        print("turn right!")
        CD.speed = 0.0
        CD.omega = -0.1
        CD.radius = 2.0
    elif action == 4:
        print("yuandi left")
        CD.speed = 0.0
        CD.omega = 0.2
        CD.radius = 0.0
    elif action == 5:
        print("yuandi right")
        CD.speed = 0.0
        CD.omega = -0.2
        CD.radius = 0.0
    elif action == 6:
        print("backward")
        CD.speed = -0.1
        CD.omega = 0.0
        CD.radius = 0.0


class state_register(object):
    """
  Store the positional information of the last frame and the current frame
  The current frame can be called from the last index of the two parts of the register
  The objective of using the register is to store the computed state of the few previous frames
  so that it can reduce some calculation
  """

    def __init__(self, register_length: int = 3):
        self.register_length = register_length
        # state is the user coordinate with (x,y)
        self.state_part = np.zeros((self.register_length, 2))
        # distance is used to calculate the reward
        self.distance_part = np.zeros((self.register_length, 1))

    # update the register of state part and the distance part
    def update(self,
               new_state_x: float,
               new_state_y: float,
               new_distance: float):
        """
      Move forward the state and the distance register buffer
      and put the new state and new distance at the end of the buffer.
      """
        self.state_part[0:self.register_length - 1, :] = self.state_part[1:self.register_length, :]
        self.state_part[self.register_length - 1, :] = np.ndarray([new_state_x, new_state_y])

        self.distance_part[0:self.register_length - 1] = self.distance_part[1:self.register_length]
        self.distance_part[self.register_length - 1] = new_distance


UState = state_register(register_length=register_length)


class data_register(object):

    def __init__(self, Camera: IRCamera, Skin: SoftSkin, win_width: int,
                 UP: PP.User_Postition_Estimate, US: state_register,
                 ir_data_width: int = 768, skin_data_width: int = 32, Normalize: bool = True):
        """The data object is to store the data buffer and can be activated and called by other function"""
        self.Camera = Camera
        self.Skin = Skin
        self.UP = UP
        self.US = US
        self.win_width = win_width
        self.ir_data_width = ir_data_width
        self.skin_data_width = skin_data_width
        self.Normalize = Normalize
        """buffer is a sequence of several consecutive frames of data"""
        self.data_buffer = np.zeros((win_width * (ir_data_width + skin_data_width), 1))
        self.new_frame = np.zeros((24, 32))

    def reading_data(self):
        """
      the reading data is to get new frame of thermal camera data and the soft skin data
      The two kinds of data will then be concatenated together
      The data buffer will be updated like a queue, the new frame will be pushed at the end of the data buffer
      """
        while True:
            self.Camera.get_irdata_once()
            if len(self.Camera.temperature) == self.ir_data_width:
                break
        """get the new frame data"""
        ir_data = np.array(self.Camera.temperature).reshape((self.ir_data_width, 1))
        if self.Normalize:
            ir_threshold = max(ir_data.mean() + 1.8, 24)
            ir_data[ir_data <= ir_threshold] = 0
            ir_data[ir_data > ir_threshold] = 1
            # normalized_temperature = (normalized_temperature-min_ir)/(max_ir-min_ir)
        """the self.new_frame will be used to calculate the state"""
        self.new_frame = np.copy(ir_data).reshape((24, 32))
        skin_data = np.array(self.Skin.temp_data).reshape((self.skin_data_width, 1))
        new_frame_data = np.concatenate((ir_data, skin_data), axis=0)
        one_frame_width = self.ir_data_width + self.skin_data_width
        self.data_buffer[0:(self.win_width - 1) * one_frame_width, 0] = self.data_buffer[
                                                                        one_frame_width:self.win_width * one_frame_width,
                                                                        0]
        self.data_buffer[(self.win_width - 1) * one_frame_width:self.win_width * one_frame_width] = new_frame_data
        return self.data_buffer


DATA = data_register(Camera=Camera, Skin=skin, win_width=win_width, UP=UserPosition, US=UState)

"""Actor-Critic"""


class ActorCritic(tf.keras.Model):
    """store the model of actor and critic"""

    def __init__(self, FFL_model: FFL):
        super().__init__()
        self.actor_critic_model = FFL_model.model

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.actor_critic_model(inputs)


AC_model = ActorCritic(FFL_model)

"""state is represented by the position of the user's position"""


def update_reward_state(DATA: data_register) -> int:
    """calculate the reward and new state(user position)"""

    """update the new state and its corresponding distance to the original point"""
    DATA.UP.get_new_img(DATA.new_frame)
    DATA.UP.get_COM(show=False)
    new_state = np.array([DATA.UP.user_x, DATA.UP.user_y]).reshape((1, 2))
    new_distance = np.linalg.norm(new_state)
    DATA.US.update(new_state_x=DATA.UP.user_x,
                   new_state_y=DATA.UP.user_y,
                   new_distance=new_distance)

    """calculate the reward"""
    """threshold is set to see whether user is in a small range of the walker center"""
    threshold = 10
    change_epsilon = 10
    break_threshold = 100
    """compare the change between the distance of the latest frame[-1] and the oldest frame[0]"""
    distance_change = DATA.US.distance_part[-1] - DATA.US.distance_part[0]
    reward = 0
    if DATA.US.distance_part[-1] < threshold:
        """means user is very close to the center of the walker"""
        reward = 2
    elif DATA.US.distance_part[-1] > threshold:
        """means the user is too far from the walker"""
        """the training episode should be stopped according to the large penalty"""
        reward = -5
    elif distance_change < -change_epsilon:
        """means user is closer to the center comparing to last frame"""
        reward = 1
    elif distance_change < change_epsilon:
        """means """
        reward = 0
    elif distance_change >= change_epsilon:
        reward = -1
    return reward


def tf_update_reward_state(DATA: data_register) -> tf.int32:
    return tf.numpy_function(update_reward_state,
                             DATA,
                             tf.int32)


def my_episode(model: FFL,
               max_steps: int,
               DATA: data_register,
               CD: ControlDriver) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    smallest_reward_threshold = -5
    for t in tf.range(max_steps):
        DATA.reading_data()
        input_data = DATA.data_buffer.reshape((-1, DATA.data_buffer.shape[0], 1))
        action_probs_t, value = model(input_data)
        action = tf.random.categorical(action_probs_t, 1)[0, 0]
        walker_driver(action, CD)

        values = values.write(t, tf.squeeze(value))
        action_probs = action_probs.write(t, action_probs_t[0, action])

        reward = tf_update_reward_state(DATA=DATA)
        rewards = rewards.write(t, reward)
        if reward <= smallest_reward_threshold:
            walker_driver(0, CD)
            break
    walker_driver(0, CD)
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probs, values, rewards


# standardize parameter
eps = np.finfo(np.float32).eps.item()


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """get the G value"""
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               gamma: float,
               max_steps: int,
               DATA: data_register) -> tf.Tensor:
    with tf.GradientTape() as tape:
        action_probs, values, rewards = my_episode(model=model,
                                                   max_steps=max_steps,
                                                   DATA=DATA,
                                                   CD=CD)

        returns = get_expected_return(rewards=rewards,
                                      gamma=gamma,
                                      standardize=True)

        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

for i in range(min_episodes_criterion):
    while True:
        a = input("Press y/Y to start the training:")
        if a == 'y' or a == 'Y':
            for i in range(3):
                print("New training iteration will start in %d seconds:\r" % (3 - i))
                time.sleep(1)
            break
        else:
            time.sleep(1)
    episode_reward = train_step(model=FFL_model, optimizer=optimizer, gamma=gamma,
                                max_steps=max_steps_per_episode, DATA=DATA)
    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
