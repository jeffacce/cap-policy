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

from real2sim.utils import add_object_to_scene, world_to_pixel, world_to_camera, action_tensor_to_matrix
from real2sim.rum_sim import RUMSim
from real2sim.controllers import PID

class SimPick(RUMSim):
    def __init__(
        self,
        scene_path: str,
        objects: [],
        secondary_objects: dict = {},
        add_secondary_object: bool = True,
        max_steps: int = 30,
        render_mode: str = None,
        reward_mode: str = "sparse",
        x_range: list = [-0.16, 0.16],
        y_range: list = [0.12, 0.35],
        z_range: list = [0.9, 0.9],
        gripper_threshold: float = 0.6,
        carry_threshold: float = 0.02,
        seed: int = 0,
    ):
        super().__init__(render_mode=render_mode, reward_mode=reward_mode, seed=seed)
        assert os.path.exists(scene_path)
        
        self.scene_path = scene_path
        self.objects = objects if objects is not None else []
        self.secondary_objects = secondary_objects if secondary_objects is not None else {}
        self.add_secondary_object = add_secondary_object
        self.carry_threshold = carry_threshold
        self.gripper_threshold = gripper_threshold
        self.max_steps = max_steps
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.object_index = -1
        self.object_name = None
        self.object_2d_position = None
        self.object_3d_position = None
        self.render_width = 256
        self.render_height = 256
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = Dict({
            "rgb": Box(low=0, high=255, shape=(self.render_height, self.render_width, 3), dtype=np.uint8),
            "object_2d_position": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "object_3d_position": Box(low=-100, high=100, shape=(3,), dtype=np.float32),
        })

    def _prepare_scene_xml(self, base_scene_path):
        with open(base_scene_path, "r") as f:
            scene_xml = f.read()
        self.object_index = (self.object_index + 1) % len(self.objects)
        object_name, object_path = self.objects[self.object_index]
        scene_xml = add_object_to_scene(scene_xml, object_path)
        secondary_object_name = None
        if self.add_secondary_object:
            filtered_secondary_objects = [obj for obj in self.secondary_objects if obj[0] != object_name]
            if len(filtered_secondary_objects) < 1:
                raise ValueError("No secondary objects available that are not the same as the main object.")
            secondary_object_name, secondary_object_path = filtered_secondary_objects[self.np_random.integers(len(filtered_secondary_objects))]
            scene_xml = add_object_to_scene(scene_xml, secondary_object_path)
        return scene_xml, object_name, secondary_object_name

    def _randomize_object_position(self, object_name, secondary=False):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{object_name}_object")
        bottom_site_position = self._get_pos_rot(f"{object_name}_bottom_site", "site")[0][2]
        center_site_position = self._get_pos_rot(f"{object_name}_object", "body")[0][2]
        bottom_to_center_distance = center_site_position - bottom_site_position
        z = bottom_to_center_distance + 0.765
        pos = [self.np_random.uniform(*self.x_range), self.np_random.uniform(*self.y_range), z]
        if secondary:
            for _ in range(100):
                if np.linalg.norm(np.array(pos[:2]) - np.array(self.initial_object_pos[:2])) >= 0.15:
                    break
                pos = [np.random.uniform(*self.x_range), np.random.uniform(*self.y_range), z]
        random_z_rotation = self.np_random.uniform(0, 2 * np.pi)
        self.model.body_quat[body_id] = R.from_euler("x", random_z_rotation, degrees=False).as_quat()
        self.model.body_pos[body_id] = pos
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.initial_rot = self._get_pos_rot(f"end_effector", "site")[1]
        return self._get_pos_rot(f"{object_name}_object", "body")[0]

    def _get_obs(self):
        rgb_obs = self._get_camera_view("egocentric")
        self.object_2d_position = world_to_pixel(
            self._get_pos_rot(f"{self.object_name}_object", "body")[0],
            self.model,
            self.data,
            "egocentric",
            480,
            360,
        )
        self.object_2d_position = np.array(self.object_2d_position, dtype=np.float32)
        self.object_3d_position = world_to_camera(
            self._get_pos_rot(f"{self.object_name}_object", "body")[0],
            self.model,
            self.data,
            "egocentric",
        )
        self.object_3d_position = np.array(self.object_3d_position, dtype=np.float32)
        return {"rgb": rgb_obs, "object_2d_position": self.object_2d_position, "object_3d_position": self.object_3d_position}

    def _compute_reward(self) -> float:
        if self.reward_mode == "dense":
            eef_position = self._get_pos_rot("end_effector", "site")[0]
            obj_position = self._get_pos_rot(f"{self.object_name}_object", "body")[0]
            dist_to_object = np.linalg.norm(eef_position - obj_position)
            close_reward = np.exp(-5.0 * dist_to_object)
            current_object_height = obj_position[2]
            initial_object_height = self.initial_object_pos[2]
            lift_distance = current_object_height - initial_object_height
            lift_reward = max(0.0, lift_distance) * 50.0
            reward = close_reward + lift_reward
        elif self.reward_mode == "sparse":
            current_object_height = self._get_pos_rot(f"{self.object_name}_object", "body")[0][2]
            initial_object_height = self.initial_object_pos[2]
            lift_distance = current_object_height - initial_object_height
            reward = 1.0 if lift_distance > self.carry_threshold else 0.0
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        scene_path, object_name, secondary_object_name = self._prepare_scene_xml(self.scene_path)
        self.object_name = object_name
        self.secondary_object_name = secondary_object_name
        self._setup_mujoco(scene_path)
        self.pid_p = PID(Kp=20.0, Ki=0.05, Kd=0.1, dt=self.model.opt.timestep)
        self.pid_r = PID(Kp=20.0, Ki=0.05, Kd=0.1, dt=self.model.opt.timestep)
        self.first_grasp = True
        mujoco.mj_step(self.model, self.data)
        self.initial_object_pos = self._randomize_object_position(object_name, secondary=False)
        if secondary_object_name is not None:
            self._randomize_object_position(secondary_object_name, secondary=True)
        self.step_count = 0
        observation = self._get_obs()
        info = {"object_name": self.object_name}
        return observation, info

    def step(self, action):
        self.step_count += 1
        delta_pose_vec = action[:6]
        grip_action = action[6]
        delta_pose_mat = action_tensor_to_matrix(delta_pose_vec, "euler")
        transformation = np.eye(4)
        transformation[:3, :3] = R.from_euler("x", -15, degrees=True).as_matrix()
        delta_pose_mat = transformation @ delta_pose_mat @ transformation.T
        current_eef_pose = self._get_pose("end_effector", "site")
        goal_pose = current_eef_pose @ delta_pose_mat
        position_error, rotation_error = self._pose_control(goal_pose)
        mujoco.mj_step(self.model, self.data)
        for _ in range(600):
            new_position_error, new_rotation_error = self._pose_control(goal_pose)
            position_error = new_position_error
            rotation_error = new_rotation_error
            mujoco.mj_step(self.model, self.data)
        if grip_action < self.gripper_threshold:
            self.pid_p.Kp = 10
            self.pid_r.Kp = 10
            self.data.ctrl[self.model.actuator("fingers_actuator").id] = -0.3
            if self.first_grasp:
                self.first_grasp = False
                object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.object_name}_object")
                self.model.body_gravcomp[object_id] = 1
                for _ in range(2000):
                    mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = bool(reward)
        truncated = bool(self.step_count >= self.max_steps)
        info = {"object_name": self.object_name, "step_count": self.step_count}
        return obs, reward, terminated, truncated, info