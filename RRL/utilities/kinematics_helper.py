import torch
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg

def gripper_real2sim(real_gripper_status: float) -> float:
    """
    Maps the real robot's gripper status (800 open, 0 closed) to the simulator's (0.0 open, 0.84 closed)
    using a quadratic function.

    Args:
        real_gripper_status (float): Gripper status from the real robot (range: 800 to 0).

    Returns:
        float: Mapped gripper status for the simulator (range: 0.0 to 0.84).
    """
    return (800 - real_gripper_status) / 800
