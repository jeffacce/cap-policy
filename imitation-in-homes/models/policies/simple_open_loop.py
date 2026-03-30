import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from quaternion import (
    as_rotation_matrix,
    quaternion,
)
import torch
P = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

def apply_permutation_transform(matrix):
    return P @ matrix @ P.T

class SimpleReplay:
    def __init__(self, poses_file_path, timeskip=8):
        self.poses_file_path = poses_file_path
        self.timeskip = timeskip
        self.idx = 0
        self.transforms = None

        self.process_poses()

    def to(self, device):
        pass

    def eval(self):
        pass

    def get_poses(self):
        with open(self.poses_file_path, "r") as f:
            lines = f.readlines()
        poses, timestamps = [], []
        for line in lines:
            line_list = eval(line)
            ts = int(line_list[0].split("<")[1].split(">")[0])
            pose = np.array(line_list[1:])
            timestamps.append(ts)
            poses.append(pose)
        timestamps = np.array(timestamps)
        poses = np.array(poses)

        return poses, timestamps

    def process_poses(self):
        quaternions = []
        translations = []
        init_pose = None

        poses, timestamps = self.get_poses()
        for pose in poses:
            qx, qy, qz, qw, tx, ty, tz = pose
            ext_matrix = np.eye(4)
            ext_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
            ext_matrix[:3, 3] = tx, ty, tz

            if init_pose is None:
                init_pose = np.copy(ext_matrix)
            relative_pose = np.linalg.inv(init_pose) @ ext_matrix

            relative_pose = apply_permutation_transform(relative_pose)
            translations.append(relative_pose[:3, -1])
            quaternions.append(
                R.from_matrix(relative_pose[:3, :3]).as_quat()
            )
        quats = np.array(quaternions)
        translations = np.array(translations)
        transforms = np.concatenate([translations, quats], axis=1)

        self.transforms = transforms
    
    def get_action(self, idx):
        prior_translations, prior_rotations = self.transforms[idx, :3], self.transforms[idx, 3:]
        next_translations, next_rotations = self.transforms[idx + self.timeskip, :3], self.transforms[idx + self.timeskip, 3:]
        # Now, create the matrices.
        prior_rot_matrices, next_rot_matrices = (
            R.from_quat(prior_rotations).as_matrix(),
            R.from_quat(next_rotations).as_matrix(),
        )
        # Now, compute the relative matrices.
        prior_matrices = np.eye(4)
        prior_matrices[:3, :3] = prior_rot_matrices
        prior_matrices[:3, 3] = prior_translations

        next_matrices = np.eye(4)
        next_matrices[:3, :3] = next_rot_matrices
        next_matrices[:3, 3] = next_translations

        relative_transforms = np.matmul(np.linalg.inv(prior_matrices), next_matrices)
        relative_translations = relative_transforms[:3, 3]
        relative_rotations = R.from_matrix(relative_transforms[:3, :3]).as_rotvec()

        gripper = 1.0

        return np.concatenate([relative_translations, relative_rotations, [gripper]], dtype=np.float32)
    
    def step(self, img, step_no):
        start_idx = self.timeskip * step_no
        if start_idx + self.timeskip >= len(self.transforms):
            print("INDEX OUT OF BOUNDS")
            action_tensor = torch.zeros(7)
            action_tensor[-1] = 1
            return action_tensor, {}
        
        action_tensor = self.get_action(start_idx)
        return torch.tensor(action_tensor), {}
    
    def reset(self):
        pass