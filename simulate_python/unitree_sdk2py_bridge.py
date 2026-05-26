import mujoco
import numpy as np
import pygame
import sys
import struct
from mujoco_lidar import MjLidarWrapper, scan_gen
from mujoco import mj_copyData, MjData

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointField_Constants_PointCloud2_ as PointCloud2_default
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import *

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
TOPIC_LIDAR = "rt/utlidar/cloud"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data, locker=None):
        self.mj_model = mj_model
        self.mj_data = mj_data


        self.FORCE_SYNC = False
        if locker is not None:
            self.locker = locker
            self.FORCE_SYNC = True

        
        self.last_lidar_time = mj_data.time
        self.lidar_copy_of_mj_data = MjData(mj_model)

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = (self.num_motor > NUM_MOTOR_IDL_GO) # 0: unitree_go, 1: unitree_hg

        self.joystick = None

        # timesteps
        self.base_frequency = int(1/self.dt)
        assert self.base_frequency % 500 == 0, "Base frequency (SIMULATE_DT in s in the config.py) must be a multiple of 500Hz (at least 0.02s)"

        self.timings = { # [frequency in Hz, tracker (always 0), function]
            "low_state": [500, 0, self.PublishLowState],
            "high_state": [500, 0, self.PublishHighState],
            "wireless_controller": [100, 0, self.PublishWirelessController],
            "lidar": [10, 0, self.PublishLidarData]
        }
        self.timings_converted = [[int(self.base_frequency/x[0]), x[1], x[2]] for x in self.timings.values()]

        # Check sensor
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        #self.lowStateThread = RecurrentThread(
        #    interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        #)

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        #self.HighStateThread = RecurrentThread(
        #    interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        #)

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        #self.WirelessControllerThread = RecurrentThread(
        #    interval=0.01,
        #    target=self.PublishWirelessController,
        #    name="sim_wireless_controller",
        #)

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        # lidar setup and publisher        
         
        lidar_type = "mid360" # the default value of the copied example.
        # needs investigation into what actually fits the real lidar on the Go2
        # this is copied from https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/examples/unitree_go2.py
        self.dynamic_lidar = False

        if lidar_type == "airy":
            self.rays_theta, self.rays_phi = scan_gen.generate_airy96()
        elif lidar_type == "mid360":
            self.livox_generator = scan_gen.LivoxGenerator(lidar_type)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.dynamic_lidar = True
        self.stand = False # default from copied example

        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[3:] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(
            mj_model,
            site_name="lidar",
            backend="cpu", # for now cpu only, maybe gpu one day
            args={"bodyexclude": mj_model.body("base_link").id, "geomgroup": geomgroup},
        )

        self.lidar_data = PointCloud2_default()
        # mostly copied from https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/examples/unitree_go2_ros2.py
        fields = [ # https://docs.ros2.org/latest/api/sensor_msgs/msg/PointField.html
            PointField_(name="x", offset=0, datatype=7, count=1),
            PointField_(name="y", offset=4, datatype=7, count=1),
            PointField_(name="z", offset=8, datatype=7, count=1),
        ]
        #for i, field in enumerate(fields):
        #    raise ValueError(self.lidar_data.fields, len(self.lidar_data.fields))
        #    self.lidar_data.fields[i] = field
        self.lidar_data.fields = fields
        self.lidar_data.header.frame_id = "lidar"
        self.lidar_data.is_bigendian = False
        self.lidar_data.point_step = 12
        self.lidar_data.height = 1
        self.lidar_data.is_dense = True

        self.lidar_data_puber = ChannelPublisher(TOPIC_LIDAR, PointCloud2_)
        self.lidar_data_puber.Init()
        #self.LidarDataThread = RecurrentThread(
        #    interval=self.lidar_dt, target=self.PublishLidarData, name="sim_lidar"
        #)
        #self.LidarDataThread.Start()

        #if not self.slower_than_real:
        #    self.LidarDataThread.Start()
        #    self.WirelessControllerThread.Start()
        #    self.HighStateThread.Start()
        #    self.lowStateThread.Start()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

        print("Finished init")

    def LowCmdHandler(self, msg: LowCmd_):
        """Update the torque applied to each motor using a LowCmd

        Implements a PD controller.

        Args:
            msg (LowCmd_): Command to be executed

        Todo:
            - The PD controller should be decoupled from the LowCmdHandler and updated at 500Hz to match the real robot
            - Currently, not sending LowCmds at 500Hz causes incorrect behaviour as the torque stays constant
        """
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = (
                    msg.motor_cmd[i].tau
                    + msg.motor_cmd[i].kp
                    * (msg.motor_cmd[i].q - self.mj_data.sensordata[i])
                    + msg.motor_cmd[i].kd
                    * (
                        msg.motor_cmd[i].dq
                        - self.mj_data.sensordata[i + self.num_motor]
                    )
                )

    def PublishLowState(self):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                    i + self.num_motor
                ]
                self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                    i + 2 * self.num_motor
                ]

            if self.have_frame_sensor_:

                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 3
                ]

                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 6
                ]

                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 9
                ]

            if self.joystick != None:
                pygame.event.get()
                # Buttons
                self.low_state.wireless_remote[2] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                0,
                                0,
                                int(self.joystick.get_axis(self.axis_id["LT"]) > 0),
                                int(self.joystick.get_axis(self.axis_id["RT"]) > 0),
                                int(self.joystick.get_button(self.button_id["SELECT"])),
                                int(self.joystick.get_button(self.button_id["START"])),
                                int(self.joystick.get_button(self.button_id["LB"])),
                                int(self.joystick.get_button(self.button_id["RB"])),
                            ]
                        ]
                    ),
                    2,
                )
                self.low_state.wireless_remote[3] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                int(self.joystick.get_hat(0)[0] < 0),  # left
                                int(self.joystick.get_hat(0)[1] < 0),  # down
                                int(self.joystick.get_hat(0)[0] > 0), # right
                                int(self.joystick.get_hat(0)[1] > 0),    # up
                                int(self.joystick.get_button(self.button_id["Y"])),     # Y
                                int(self.joystick.get_button(self.button_id["X"])),     # X
                                int(self.joystick.get_button(self.button_id["B"])),     # B
                                int(self.joystick.get_button(self.button_id["A"])),     # A
                            ]
                        ]
                    ),
                    2,
                )
                # Axes
                sticks = [
                    self.joystick.get_axis(self.axis_id["LX"]),
                    self.joystick.get_axis(self.axis_id["RX"]),
                    -self.joystick.get_axis(self.axis_id["RY"]),
                    -self.joystick.get_axis(self.axis_id["LY"]),
                ]
                packs = list(map(lambda x: struct.pack("f", x), sticks))
                self.low_state.wireless_remote[4:8] = packs[0]
                self.low_state.wireless_remote[8:12] = packs[1]
                self.low_state.wireless_remote[12:16] = packs[2]
                self.low_state.wireless_remote[20:24] = packs[3]

            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):

        if self.mj_data != None:
            self.high_state.position[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 10
            ]
            self.high_state.position[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 11
            ]
            self.high_state.position[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 12
            ]

            self.high_state.velocity[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 13
            ]
            self.high_state.velocity[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 14
            ]
            self.high_state.velocity[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 15
            ]

        self.high_state_puber.Write(self.high_state)

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PublishLidarData(self):
        """

        Todo:
            - Add simulation time as timestamp
        """

        if self.FORCE_SYNC:
            self.locker.acquire()
        mj_copyData(self.lidar_copy_of_mj_data, self.mj_model, self.mj_data)
        #mjd_copy = self.mj_data.mj_copyData(self.mj_model)
        if self.FORCE_SYNC:
            self.locker.release()
        
        self.lidar.trace_rays(self.lidar_copy_of_mj_data, self.rays_theta, self.rays_phi)
        
        #self.lidar.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
        points = self.lidar.get_hit_points()
        #raise ValueError(points, self.lidar)

        time_stamp = Time_(0,0) # one day seconds, nanoseconds (int32). What is the time reference?

        self.lidar_data.header.stamp = time_stamp
        self.lidar_data.row_step = self.lidar_data.point_step * points.shape[0]
        self.lidar_data.width = points.shape[0]
        self.lidar_data.data = points.tobytes()

        self.lidar_data_puber.Write(self.lidar_data)

    def MuJoCo_timestep(self):
        for subject in self.timings_converted:
            subject[1] += 1
            if subject[0] <= subject[1]:
                subject[2]()
                subject[1] = 0



    def PrintSceneInformation(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self.mj_model.sensor_dim[i],
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
