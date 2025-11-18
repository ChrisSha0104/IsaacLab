import multiprocess as mp
import numpy as np
import copy

from agents.gello_agent import GelloAgent
from agents.gello_agent import DynamixelRobotConfig
from dynamixel.driver import DynamixelDriver


class GelloListener(mp.Process):

    def __init__(
        self, 
        gello_port: str = '/dev/ttyUSB0',
    ):
        super().__init__()

        self.num_joints = 7
        self.gello_port = gello_port
        self.do_offset_calibration = False  # hardcoded: whether to recalibrate the offset
        self.verbose = True

        examples = dict()
        examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.command = mp.Array('d', examples['command'])
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.end_wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self):
        return copy.deepcopy(np.array(self.command[:]))

    def init_gello(self):
        if self.do_offset_calibration:
            joint_offsets, gripper_config = self.calibrate_offset(port=self.gello_port)
        else:
            joint_offsets = (
                4 * np.pi / 8,
                8 * np.pi / 8,
                0 * np.pi / 8,
                4 * np.pi / 8,
                8 * np.pi / 8,
                9 * np.pi / 8,
                8 * np.pi / 8
            )
            gripper_config = (8, 288, 246)

        dynamixel_config = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=joint_offsets,
            joint_signs=(1, 1, 1, 1, 1, 1, 1),
            gripper_config=gripper_config,
        )
        gello_port = self.gello_port
        start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
        try:
            agent = GelloAgent(port=gello_port, dynamixel_config=dynamixel_config, start_joints=start_joints)
        except:
            raise RuntimeError(f"Failed to connect to Gello on port {gello_port}")
        self.agent = agent

        self.ready_event.set()
    
    def calibrate_offset(self, port):
        start_joints = tuple(np.deg2rad([0, -45, 0, 30, 0, 75, 0]))  # The joint angles that the GELLO is placed in at (in radians)
        joint_signs = (1, 1, 1, 1, 1, 1, 1)  # The joint angles that the GELLO is placed in at (in radians)

        joint_ids = list(range(1, self.num_joints + 2))
        driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

        # assume that the joint state shouold be start_joints
        # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
        # this is done by brute force, we seach in a range of +/- 8pi

        def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
            joint_sign_i = joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = start_joints[index]
            return np.abs(joint_i - start_i)

        for _ in range(10):
            driver.get_joints()  # warmup

        for _ in range(1):
            best_offsets = []
            curr_joints = driver.get_joints()
            for i in range(self.num_joints):
                best_offset = 0
                best_error = 1e6
                for offset in np.linspace(-8 * np.pi, 8 * np.pi, 16 * 8 + 1):  # intervals of pi/8
                    error = get_error(offset, i, curr_joints)
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                best_offsets.append(best_offset)

        gripper_open = np.rad2deg(driver.get_joints()[-1]) - 0.2
        gripper_close = np.rad2deg(driver.get_joints()[-1]) - 42
        if self.verbose:
            print()
            print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/8)))}*np.pi/8" for x in best_offsets])
                + " ]",
            )
            print(
                "gripper open (degrees)       ",
                gripper_open,
            )
            print(
                "gripper close (degrees)      ",
                gripper_close,
            )

        joint_offsets = tuple(best_offsets)
        gripper_config = (8, gripper_open, gripper_close)
        return joint_offsets, gripper_config

    def run(self):
        self.init_gello()

        while self.alive:
            try:
                action = self.agent.get_action()
                self.command[:] = action
            except:
                print(f"Error in GelloListener")
                break

        self.stop()
        print("GelloListener exit!")
        
    @property
    def alive(self):
        return not self.stop_event.is_set() and self.ready_event.is_set()

