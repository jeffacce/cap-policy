import os
import sys
from pathlib import Path
import pickle
import time
import numpy as np



from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import (
    get_default_controller_config,
    verify_controller_config,
)

from .utils import notify_component_start
from .network import create_response_socket
from .messages import FrankaAction, FrankaState
from .constants import (
    CONTROL_PORT,
    HOST,
    CONTROL_FREQ,
)

# Get the absolute path to the robot-server directory
current_file = Path(__file__).resolve()
robot_server_path = current_file.parent.parent.parent.parent.parent
sys.path.insert(0, str(robot_server_path))
print(f"Added to Python path: {robot_server_path}")

robot_path = robot_server_path / "robot"
sys.path.insert(0, str(robot_path))
print(f"Also added robot path: {robot_path}")

CONFIG_ROOT = Path(__file__).parent / "configs"


class FrankaServer:
    def __init__(self, cfg):
        self._robot = Robot(cfg, CONTROL_FREQ)
        # Action REQ/REP
        self.action_socket = create_response_socket(HOST, CONTROL_PORT)

    def init_server(self):
        # connect to robot
        print("Starting Franka server...")
        self._robot.reset_robot()
        self.control_daemon()

    def get_state(self):
        quat, pos = self._robot.last_eef_quat_and_pos
        gripper = self._robot.last_gripper_action
        if quat is not None and pos is not None and gripper is not None:
            state = FrankaState(
                pos=pos.flatten().astype(np.float32),
                quat=quat.flatten().astype(np.float32),
                gripper=gripper,
                timestamp=time.time(),
            )
            return bytes(pickle.dumps(state, protocol=-1))
        else:
            return b"state_error"

    def control_daemon(self):
        notify_component_start(component_name="Franka Control Subscriber")
        try:
            while True:
                command = self.action_socket.recv()
                if command == b"get_state":
                    self.action_socket.send(self.get_state())
                else:
                    franka_control: FrankaAction = pickle.loads(command)
                    if franka_control.reset:
                        # Extract gripper value from array if it's an array
                        gripper_value = franka_control.gripper[0] if hasattr(franka_control.gripper, '__len__') else franka_control.gripper
                        self._robot.reset_joints(gripper_open=bool(gripper_value))
                        time.sleep(1)
                    else:
                        # Extract gripper value from array if it's an array
                        gripper_value = franka_control.gripper[0] if hasattr(franka_control.gripper, '__len__') else franka_control.gripper
                        self._robot.osc_move(
                            franka_control.pos,
                            franka_control.quat,
                            gripper_value,
                        )
                    self.action_socket.send(self.get_state())
        except KeyboardInterrupt:
            pass
        finally:
            self._robot.close()
            self.action_socket.close()


class Robot(FrankaInterface):
    def __init__(self, cfg, control_freq):
        super(Robot, self).__init__(
            general_cfg_file=os.path.join(CONFIG_ROOT, cfg),
            use_visualizer=False,
            control_freq=control_freq,
        )
        self.velocity_controller_cfg = verify_controller_config(
            YamlConfig(
                os.path.join(CONFIG_ROOT, "osc-pose-controller.yml")
            ).as_easydict()
        )

    def reset_robot(self):
        self.reset()

        print("Waiting for the robot to connect...")
        while len(self._state_buffer) == 0:
            time.sleep(0.01)

        print("Franka is connected")

    def osc_move(self, target_pos, target_quat, gripper_state):
        num_steps = 3

        for _ in range(num_steps):
            target_mat = transform_utils.pose2mat(pose=(target_pos, target_quat))

            current_quat, current_pos = self.last_eef_quat_and_pos
            current_mat = transform_utils.pose2mat(
                pose=(current_pos.flatten(), current_quat.flatten())
            )

            pose_error = transform_utils.get_pose_error(
                target_pose=target_mat, current_pose=current_mat
            )

            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat

            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

            action_pos = pose_error[:3]
            action_axis_angle = axis_angle_diff.flatten()

            action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_state]

            self.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=self.velocity_controller_cfg,
            )

    def reset_joints(
        self,
        timeout=7,
        gripper_open=False,
    ):
        start_joint_pos = [
     0.1343,
    -0.6109,
    -0.2147,
    -2.8471,
    -0.0105,
     1.7977,
     0.9210
]
        assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")

        if gripper_open:
            gripper_action = -1
        else:
            gripper_action = 1

        if type(start_joint_pos) is list:
            action = start_joint_pos + [gripper_action]
        else:
            action = start_joint_pos.tolist() + [gripper_action]
        start_time = time.time()
        while True:
            if self.received_states and self.check_nonzero_configuration():
                if (
                    np.max(np.abs(np.array(self.last_q) - np.array(start_joint_pos)))
                    < 1e-3
                ):
                    break
            self.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time()

            # Add timeout
            if end_time - start_time > timeout:
                break
        return True
