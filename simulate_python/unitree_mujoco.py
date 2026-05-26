import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

def update_ui(text): 
    """Provides a nice console feedback"""
    sys.stdout.write("\r")
    sys.stdout.write(" "*15)
    sys.stdout.write("\r")
    sys.stdout.write(text)
    sys.stdout.flush()


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
unitree = UnitreeSdk2Bridge(mj_model, mj_data, locker=locker)

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujucoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=unitree.MujocoKeyCallback)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

stats_counter = 50
stats_timer = time.perf_counter()

def SimulationThread():
    global mj_data, mj_model, unitree, stats_timer, stats_counter

    if config.USE_JOYSTICK:
        #raise ValueError("ASDFGSADFGBSDAFGBSDBSDRTTB")
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        #print("asdfiakohfaspedNPSWERFNESIRsdfknsdfknsdfnsdnSDOFOKJNSADFJKNVASDFKVN")
        unitree.PrintSceneInformation()
        #raise ValueError("ASDFGSADFGBSDAFGBSDBSDRTTB")

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        unitree.MuJoCo_timestep()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        stats_counter -= 1
        if stats_counter <= 0:
            stats_counter = 50
            update_ui(f"{(50/(time.perf_counter() - stats_timer)):.4f} steps/s")
            stats_timer = time.perf_counter()

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    print("Starting threads...")
    viewer_thread.start()
    sim_thread.start()
    print("Ending main")