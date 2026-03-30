import torch
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T  # For ToTensor conversion
import time

from .utils import batch_build_affines, batch_apply_transform, batch_extract_euler_xyz

P = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)

class VectorizedBuffer:
    def __init__(self, batch_size, buffer_size, image_shape, act_dim, device):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.act_dim = act_dim
        self.device = device
        self.image_buffer = None
        self.goal_buffer = None
        self.action_buffer = None
        self.image_buffers_sizes = torch.zeros(batch_size, device=device)
        self.goal_buffer_size = torch.zeros(batch_size, device=device)
        self.action_buffers_sizes = torch.zeros(batch_size, device=device)

    def add_image(self, new_images):
        if self.image_buffer is None:
            self.image_buffer = (
                new_images.unsqueeze(1)
                .repeat(1, self.buffer_size, 1, 1, 1)
                .to(self.device)
            )
        else:
            for b in range(self.batch_size):
                if self.image_buffers_sizes[b] == 0:
                    self.image_buffer[b] = (
                        new_images[b].unsqueeze(0).repeat(self.buffer_size, 1, 1, 1)
                    )
                else:
                    self.image_buffer[b] = torch.roll(
                        self.image_buffer[b], shifts=-1, dims=0
                    )
                    self.image_buffer[b, -1] = new_images[b]
        self.image_buffers_sizes += 1
        self.image_buffers_sizes = torch.clamp(
            self.image_buffers_sizes, max=self.buffer_size
        )

    def add_goal(self, new_goals):
        if self.goal_buffer is None:
            self.goal_buffer = (
                new_goals.unsqueeze(1).repeat(1, self.buffer_size, 1).to(self.device)
            )
        else:
            for b in range(self.batch_size):
                if self.goal_buffer_size[b] == 0:
                    self.goal_buffer[b] = (
                        new_goals[b].unsqueeze(0).repeat(self.buffer_size, 1)
                    )
                else:
                    self.goal_buffer[b] = torch.roll(
                        self.goal_buffer[b], shifts=-1, dims=0
                    )
                    self.goal_buffer[b, -1] = new_goals[b]
        self.goal_buffer_size += 1
        self.goal_buffer_size = torch.clamp(self.goal_buffer_size, max=self.buffer_size)

    def reset(self, batch_indices):

        self.image_buffers_sizes[batch_indices] = 0
        self.goal_buffer_size[batch_indices] = 0
        self.action_buffers_sizes[batch_indices] = 0
        if self.image_buffer is not None:
            self.image_buffer[batch_indices] = torch.zeros_like(
                self.image_buffer[batch_indices]
            )
        if self.goal_buffer is not None:
            self.goal_buffer[batch_indices] = torch.zeros_like(
                self.goal_buffer[batch_indices]
            )
        if self.action_buffer is not None:
            self.action_buffer[batch_indices] = torch.zeros_like(
                self.action_buffer[batch_indices]
            )
            
    def add_action(self, new_actions):
        B = new_actions.shape[0]
        if self.action_buffer is None:
            self.action_buffer = torch.zeros(
                B, self.buffer_size - 1, self.act_dim, device=self.device
            )
            self.action_buffers_sizes = torch.zeros(B, device=self.device)
        for b in range(B):
            if self.action_buffers_sizes[b] != 0:
                self.action_buffer[b] = torch.roll(
                    self.action_buffer[b], shifts=-1, dims=0
                )
            self.action_buffer[b, -1] = new_actions[b]
        self.action_buffers_sizes += 1

    def get_input_sequence(self):
        B = self.image_buffer.shape[0]
        if self.action_buffer is None:
            action_buffer = torch.zeros(
                B, self.buffer_size - 1, self.act_dim, device=self.device
            )
        else:
            action_buffer = self.action_buffer
        base_act = torch.zeros(B, 1, self.act_dim, device=self.device)
        act_seq = torch.cat([action_buffer, base_act], dim=1)
        if self.goal_buffer is None:
            goal_seq = None
        else:
            goal_seq = torch.stack([goal for goal in self.goal_buffer]).to(
                dtype=torch.float32
            )
        return self.image_buffer, goal_seq, act_seq


def unwrap_model(model):
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        return model.module
    return model


class StrippedPolicy:
    def __init__(self, model, loss_fn, buffer_size, device, condition=None):

        self.to_tensor = T.ToTensor()
        self.model = unwrap_model(model)
        self.loss_fn = unwrap_model(loss_fn)
        self.buffer_size = buffer_size
        self.device = device

        valid_conditions = ("3d", "2d")
        if condition is not None and condition not in valid_conditions:
            raise ValueError(f"'condition' must be one of {valid_conditions}, got '{condition}'")
        self.condition = condition

        self.model.eval()
        self.loss_fn.eval()

        self.vectorized_buffer = None
        self.act_dim = 7

        self.rot_yx_90 = (
            R.from_euler("y", 90, degrees=True).as_matrix()
            @ R.from_euler("x", 90, degrees=True).as_matrix()
        )
        self.Tyx = np.eye(4, dtype=np.float32)
        self.Tyx[:3, :3] = self.rot_yx_90

        rot_z_90 = R.from_euler("z", 90, degrees=True).as_matrix()
        Tz = np.eye(4, dtype=np.float32)
        Tz[:3, :3] = rot_z_90

        M = self.Tyx @ Tz @ P.T
        M_inv = M.T

        self.M_t = torch.from_numpy(M).to(device)
        self.M_inv_t = torch.from_numpy(M_inv).to(device)

    def reset(self, indicies=None):
        if indicies is None:
            self.vectorized_buffer = None
        else:
            self.vectorized_buffer.reset(indicies)

    def process_image(self, img):
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        return self.to_tensor(img)

    def forward(self, obs, goal=None):

        if self.condition is not None and goal is None:
            raise ValueError("Condition is set, but goal is None.")
        if self.condition is None and goal is not None:
            raise ValueError("Condition is None, but goal is set.")

        if self.condition == "2d" and goal.shape[1] != 2:
            raise ValueError(
                f"Goal shape is {goal.shape}, but it should be (B, 2) for 2D condition."
            )
        if self.condition == "3d" and goal.shape[1] != 3:
            raise ValueError(
                f"Goal shape is {goal.shape}, but it should be (B, 3) for 3D condition."
            )

        B = obs.shape[0]

        processed_images = torch.stack(
            [self.process_image(obs[i]) for i in range(B)]
        ).to(self.device)

        if self.vectorized_buffer is None:
            image_shape = processed_images.shape[1:]
            self.vectorized_buffer = VectorizedBuffer(
                batch_size=B,
                buffer_size=self.buffer_size,
                image_shape=image_shape,
                act_dim=self.act_dim,
                device=self.device,
            )

        self.vectorized_buffer.add_image(processed_images)

        if self.condition is not None:
            processed_goals = torch.stack(
                [torch.tensor(goal[i], device=self.device) for i in range(B)]
            )
            processed_goals = processed_goals.view(B, -1)
            self.vectorized_buffer.add_goal(processed_goals)

        img_seq, goal_seq, act_seq = self.vectorized_buffer.get_input_sequence()

        with torch.no_grad():
            model_input = (img_seq, goal_seq, act_seq)
            model_output = self.model(model_input)
            action_tensors, logs = self.loss_fn.step(
                model_input, model_output, return_all=True
            )

        action_tensors = action_tensors.squeeze(1).to(self.device)
        self.vectorized_buffer.add_action(action_tensors)

        affines_t = batch_build_affines(
            action_tensors, rot_unit="axis"
        )  # converting the action tensor to a homogeneous matrix
        affines_t = batch_apply_transform(
            affines_t, self.M_t, self.M_inv_t
        )  # applying the transformations to go from from app to camera axis (as in controller.py)

        final_translations = affines_t[:, :3, 3]
        final_rotations = affines_t[:, :3, :3]
        final_eulers = batch_extract_euler_xyz(final_rotations)
        final_grasp = action_tensors[:, 6].unsqueeze(1)

        final_actions = (
            torch.cat([final_translations, final_eulers, final_grasp], dim=1)
            .cpu()
            .numpy()
        )

        return final_actions