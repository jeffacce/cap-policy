import os
import platform

if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"
elif platform.system() == "Linux":
    os.environ["MUJOCO_GL"] = "osmesa"
    
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
import mujoco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class RUMSim(gym.Env):
    metadata = {
        "render_modes": ["human", "human_goal", "rgb_array"],
        "reward_modes": ["dense", "sparse"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, reward_mode="sparse", seed: int = 0):
        super().__init__()
        assert render_mode in self.metadata["render_modes"] or (render_mode is None)
        assert reward_mode in self.metadata["reward_modes"] or (reward_mode is None)
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.np_random, _ = seeding.np_random(seed)
        self.render_width = 256
        self.render_height = 256
        self.model = None
        self.data = None
        self.rgb_renderer = None
        self.pid_p = None
        self.pid_r = None
        self.gripper_actuator_names = [
            "gripper_joint_x",
            "gripper_joint_y",
            "gripper_joint_z",
            "gripper_joint_rx",
            "gripper_joint_ry",
            "gripper_joint_rz",
        ]

    def _setup_mujoco(self, scene_xml: str):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            del self.data
            del self.rgb_renderer
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.gripper_actuator_ids = [self.model.actuator(name).id for name in self.gripper_actuator_names]
        self.rgb_renderer = mujoco.Renderer(self.model, height=360, width=480)

    def _get_camera_view(self, camera_name: str):
        self.rgb_renderer.update_scene(self.data, camera_name)
        rgb = self.rgb_renderer.render()
        rgb = cv2.resize(rgb, (self.render_width, self.render_height))
        return rgb

    def _get_errors(self, goal_pose, current_pose):
        position_error = self.initial_rot @ (goal_pose[:3, 3] - current_pose[:3, 3])
        rotation_error = self.initial_rot.T @ (R.from_matrix(goal_pose[:3, :3] @ current_pose[:3, :3].T).as_rotvec())
        return position_error, rotation_error

    def _pose_control(self, goal_pose):
        current_pose = self._get_pose("end_effector", "site")
        position_error, rotation_error = self._get_errors(goal_pose, current_pose)
        position_signal = self.pid_p(position_error)
        rotation_signal = self.pid_r(rotation_error)
        signals = np.concatenate((position_signal, rotation_signal))
        self.data.ctrl[self.gripper_actuator_ids] = signals
        return position_error, rotation_error

    def render(self):
        if self.render_mode is None:
            return
        rgb_ego = self._get_camera_view("egocentric")
        if hasattr(self, "object_2d_position") and self.render_mode == "human_goal":
            object_pixel = (
                min(int(self.object_2d_position[0] * self.render_width), self.render_width - 1),
                min(int(self.object_2d_position[1] * self.render_height), self.render_height - 1),
            )
            rgb_ego = cv2.drawMarker(rgb_ego, object_pixel, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        rgb_top = self._get_camera_view("top_view")
        rgb_stack = np.concatenate((rgb_ego, rgb_top), axis=1)
        if self.render_mode in ["human", "human_goal"]:
            cv2.imshow("RUMSimEnvironment", cv2.cvtColor(rgb_stack, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return None
        elif self.render_mode == "rgb_array":
            return rgb_stack

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
    
    def _get_pos_rot(self, name, type_):
        type_lower = type_.lower()
        if type_lower == "body":
            pos = self.data.body(name).xpos.copy()
            rot_mat = self.data.body(name).xmat.copy().reshape((3, 3))
            return pos, rot_mat
        elif type_lower == "site":
            pos = self.data.site(name).xpos.copy()
            rot_mat = self.data.site(name).xmat.copy().reshape((3, 3))
            return pos, rot_mat
        elif type_lower == "geom":
            pos = self.data.geom(name).xpos.copy()
            rot_mat = self.data.geom(name).xmat.copy().reshape((3, 3))
            return pos, rot_mat
        else:
            raise ValueError(f'type_ must be one of "geom", "body", or "site", got "{type_}".')

    def _get_pose(self, name, type_):
        pos, rot_mat = self._get_pos_rot(name, type_)
        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = rot_mat
        return pose