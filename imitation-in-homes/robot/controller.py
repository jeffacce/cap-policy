from enum import Enum
import logging
import time
from typing import Dict
import signal
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt

from utils.action_transforms import invert_permutation_transform, Z90
from utils.zmq_utils import ZMQKeypointSubscriber, ZMQCameraSubscriber
from utils.rpc import RPCClient
from .utils import (
    AsyncImageActionSaver,
    AsyncImageDepthActionSaver,
    ImageActionBufferManager,
    ImageDepthActionBufferManager,
    ImageActionGoalBufferManager,
    schedule_init,
)
from .object_tracking import get_transformation_matrix, get_new_2d_point
logger = logging.getLogger(__name__)
GRIPPER_FROM_CAMERA = [0.00,0.18,0.04]

schedule = None
LOCALHOST = "127.0.0.1"
ANYCAST = "0.0.0.0"
task_to_params_dict = {
    "pick": {
        "closing_threshold": 0.7,
        "continuous_gripper": False,
        "depth_offset": 0.03,
    },
    "open": {
        "closing_threshold": 0.1,
        "continuous_gripper": True,
        "depth_offset": 0.00,
    },
    "close": {
        "closing_threshold": 0.2,
        "continuous_gripper": True,
        "depth_offset": 0.00,
    },
}


class StartingPositions(Enum):
    HOME = 0
    POSITION_1 = 1
    POSITION_2 = 2
    POSITION_3 = 3
    POSITION_4 = 4
    POSITION_5 = 5
    POSITION_6 = 6
    POSITION_7 = 7
    POSITION_8 = 8
    POSITION_9 = 9
    POSITION_10 = 10


class Controller:
    def __init__(self, cfg=None):
        global schedule

        self.cfg = cfg
        self.task = cfg["task"]
        self.closing_threshold = task_to_params_dict[self.task]["closing_threshold"]
        self.continuous_gripper = task_to_params_dict[self.task]["continuous_gripper"]
        self.use_depth = cfg["use_depth"]
        self.stream_depth = cfg["stream_depth"]
        self.use_pose = cfg["use_pose"]
        self.goal_dim = cfg["goal_dim"]
        self.use_goals = cfg["goal_dim"] > 0

        network_cfg: Dict = cfg["network"]
        self.robot = RPCClient(network_cfg.get("remote"), network_cfg["action_port"])
        subscriber = ZMQCameraSubscriber(
            network_cfg.get("remote", LOCALHOST),
            network_cfg["camera_port"],
            network_cfg.get("mode", "RGB" if not self.use_depth and not self.stream_depth else "RGBD"),
        )
        self.subscriber = subscriber

        if not self.use_depth:
            self.async_saver = AsyncImageActionSaver(cfg["image_save_dir"])
        else:
            self.async_saver = AsyncImageDepthActionSaver(cfg["image_save_dir"])

        if self.use_pose:
            self.pose_subscriber = ZMQKeypointSubscriber(
                network_cfg.get("remote", LOCALHOST),
                network_cfg["pose_port"],
                "pose"
            )

        self.image_action_buffer_manager = self.create_buffer_manager()

        self.device = cfg["device"]
        schedule = schedule_init(
            self,
            max_h=cfg["robot_params"]["max_h"],
            max_base=cfg["robot_params"]["max_base"],
        )

        self.run_n = -1
        self.step_n = 0
        self.schedul_no = 0
        self.h = cfg["robot_params"]["h"]
        self.gt = self.robot.STRETCH_GRIPPER_TIGHT[0]
        self.depth_offset = task_to_params_dict[self.task]["depth_offset"]

        self.abs_gripper = cfg["robot_params"]["abs_gripper"]
        self.gripper = 1.0
        self.rot_unit = cfg["robot_params"]["rot_unit"]

        if cfg.get("goal_conditional") is True:
            import sentence_transformers

            model = sentence_transformers.SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._goal_conditional = True
            self._embedding = model.encode([cfg["goal_description"]])
            self._embedding = torch.tensor(self._embedding).to(self.device)
            del model
        else:
            self._goal_conditional = False

        self._should_home = False
        self._interrupted = False
        self.object_name = None

    def setup_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def create_buffer_manager(self):
        if self.use_depth:
            return ImageDepthActionBufferManager(
                self.cfg["image_buffer_size"],
                self.async_saver,
                self.cfg["dataset"]["train"]["config"].get("depth_cfg"),
            )
        elif self.use_goals:
            return ImageActionGoalBufferManager(
                self.cfg["image_buffer_size"],
                self.async_saver,
            )
        else:
            return ImageActionBufferManager(
                self.cfg["image_buffer_size"], self.async_saver
            )

    def action_tensor_to_matrix(self, action_tensor):
        affine = np.eye(4)
        if self.rot_unit == "euler":
            r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
        elif self.rot_unit == "axis":
            r = R.from_rotvec(action_tensor[3:6])
        else:
            raise NotImplementedError
        affine[:3, :3] = r.as_matrix()
        affine[:3, -1] = action_tensor[:3]

        return affine

    def matrix_to_action_tensor(self, matrix):
        r = R.from_matrix(matrix[:3, :3])
        action_tensor = np.concatenate(
            (matrix[:3, -1], r.as_euler("xyz", degrees=False))
        )
        return action_tensor

    def cam_to_robot_frame(self, matrix):
        return invert_permutation_transform(matrix)

    def _update_log_keys(self, logs):
        new_logs = {}
        for k in logs.keys():
            new_logs[k + "_" + str(self.run_n)] = logs[k]

        return new_logs
    
    @contextmanager
    def _handle_interrupt(self):
        """Context manager to handle SIGINT gracefully."""
        original_handler = signal.getsignal(signal.SIGINT)
        self._interrupted = False
        
        def signal_handler(signum, frame):
            self._interrupted = True
            self._should_home = True
            print("\nInterrupted! Returning to previous level...")
        
        if self._interrupted:
            signal.signal(signal.SIGINT, original_handler)
            self._interrupted = False
        else:
            try:
                signal.signal(signal.SIGINT, signal_handler)
                yield
            finally:
                signal.signal(signal.SIGINT, original_handler)
                self._interrupted = False

    def _run_policy(self, run_for=1):
        while run_for > 0:
            cv2_img, timestamp = self.subscriber.recv_rgb_image()
            logger.info(f"time to receive image: {time.time() - timestamp}")
            self.image_action_buffer_manager.add_image(cv2_img)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                if not self._goal_conditional:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )
                else:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        self._embedding,
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )

                action_tensor, logs = self.model.step(
                    input_tensor_sequence, step_no=self.step_n
                )
                if "indices" in logs:
                    indices = logs["indices"].squeeze()
                    for nbhr, idx in enumerate(indices):
                        img = self.model.train_dataset[idx]
                        img = (
                            (img[0][0]).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.async_saver.save_image(img, nbhr=nbhr)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)


            # create a transformation matrix for the action_robot
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R.from_euler('xyz', action_robot[3:6]).as_matrix()
            transformation_matrix[:3, 3] = action_robot[:3]

            # rotate 90 degrees about z axis
            rotation_matrix_z = R.from_euler('z', 90, degrees=True).as_matrix()
            transformation_matrix_90_z = np.eye(4)
            transformation_matrix_90_z[:3, :3] = rotation_matrix_z

            # apply the 90-degree transformation to the action_robot transformation matrix
            transformed_matrix = transformation_matrix_90_z @ transformation_matrix @ transformation_matrix_90_z.T

            # extract the updated rotation vector and translation from the transformed matrix
            updated_rotvec = R.from_matrix(transformed_matrix[:3, :3]).as_euler('xyz')
            updated_translation = transformed_matrix[:3, 3]

            # update the action_robot with the transformed values
            action_robot[:3] = updated_translation
            action_robot[3:6] = updated_rotvec

            gripper = action_tensor[-1]
            logger.info(f"Gripper: {gripper} {self.abs_gripper} {self.gripper}")

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper
            else:
                # Update member variable to be deplayed in the UI
                self.gripper = gripper

            logger.info("calling move_to_pose")
            self.robot.move_to_pose(action_robot[:3], action_robot[3:6], gripper)

            run_for -= 1
            self.step_n += 1

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = [x, y]
            print(f"Clicked coordinates: {self.clicked_point}")
    
    def _get_clicked_point(self, cv2_img, np_depth):
        self.clicked_point = None  # Reset before each image

        cv2.imshow("Image", cv2_img)
        cv2.setMouseCallback("Image", self.click_event)
        
        while self.clicked_point is None:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        cv2.destroyWindow("Image")

        self.object_x, self.object_y, self.object_depth = self._point2d_to_3d(self.clicked_point, np_depth)
        print(f"Object depth: {self.object_depth}")
    
    def _point2d_to_3d(self, p2d, np_depth):
        x, y = p2d[0] / 256, p2d[1] / 256
        z = np_depth[int(y * 192), int(x * 256)] + self.depth_offset
        return x, y, z

    def _run_policy_goals(self, run_for=1, hz=1):
        period = 1 / hz

        while run_for > 0 and not self._interrupted:
            last_time = time.time()
            if self.stream_depth:
                cv2_img, np_depth, timestamp = self.subscriber.recv_image_and_depth()
            else:
                cv2_img, timestamp = self.subscriber.recv_rgb_image()
            pose = None
            while pose is None:
                pose = self.pose_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            self.robot.remember_pose()
            print("time to receive image:", time.time() - timestamp)

            # cv2_img = self._apply_clahe_luma(cv2_img)

            if self.step_n == 0:
                self._get_clicked_point(cv2_img, np_depth)
                self.start_pose = pose
                self.start_pose_matrix = get_transformation_matrix(pose[4:], pose[:4])
                new_3d_point = get_new_2d_point(self.object_x, self.object_y, self.object_depth, None)
                self.clicked_point = [self.clicked_point[0] / 256, self.clicked_point[1] / 256]
            else:
                current_pose_matrix = get_transformation_matrix(pose[4:], pose[:4])
                relative_transformation_matrix = np.linalg.inv(self.start_pose_matrix) @ current_pose_matrix
                new_2d_point, new_3d_point = get_new_2d_point(self.object_x, self.object_y, self.object_depth, relative_transformation_matrix)
                self.clicked_point = [new_2d_point[0] / 960, new_2d_point[1] / 720]
                print("object point:", self.clicked_point)

                # Only plot after first frame when we have new_2d_point
                # Plot the transformed point on the image
                plot_x = int(new_2d_point[0] / 960 * 256)
                plot_y = int(new_2d_point[1] / 720 * 256)
                if "DISPLAY" in os.environ:
                    cv2_img_copy = cv2_img.copy()
                    cv2.drawMarker(cv2_img_copy, (plot_x, plot_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.imshow("Tracking", cv2_img_copy)
                    cv2.waitKey(30)
                

            x, y, z = new_3d_point
            new_3d_point = [-x, z, y] # converting from canonical frame to labels.json frame
            print("new_3d_point:", new_3d_point)
            print("clicked_point:", self.clicked_point)
            print(self.goal_dim)

            point_condition = new_3d_point if self.goal_dim == 3 else self.clicked_point
            if self.gripper <= self.closing_threshold and self.goal_dim == 3:
                point_condition = GRIPPER_FROM_CAMERA

            self.image_action_buffer_manager.add_image(cv2_img)
            if self.use_goals:
                self.image_action_buffer_manager.add_goal(np.array(point_condition))

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                if not self._goal_conditional:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )
                else:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        self._embedding,
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )

                action_tensor, logs = self.model.step(
                    input_tensor_sequence, step_no=self.step_n
                )
                if "indices" in logs:
                    indices = logs["indices"].squeeze()
                    for nbhr, idx in enumerate(indices):
                        img = self.model.train_dataset[idx]
                        img = (
                            (img[0][0]).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.async_saver.save_image(img, nbhr=nbhr)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            transformed_matrix = Z90 @ action_robot_matrix @ Z90.T
            action_robot = self.matrix_to_action_tensor(transformed_matrix)

            gripper = action_tensor[-1]
            logger.info(f"Gripper: {gripper} {self.abs_gripper} {self.gripper}")

            if self.continuous_gripper:
                gripper = gripper if gripper > self.closing_threshold else 0.0
            else:
                gripper = 1.0 if gripper > self.closing_threshold else 0.0
            
            logger.info("calling move_to_pose")
            self.robot.move_to_pose(action_robot[:3], action_robot[3:6], gripper, prev_pose=True)

            run_for -= 1
            self.step_n += 1
            if gripper < self.closing_threshold:
                time.sleep(1)
                break
            time.sleep(max(0, period - (time.time() - last_time)))

    def _run_policy_depth(self, run_for=1):
        while run_for > 0:
            cv2_img, np_depth, timestamp = self.subscriber.recv_image_and_depth()
            self.image_action_buffer_manager.add_image(cv2_img)
            self.image_action_buffer_manager.add_depth(np_depth)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                input_tensor_sequence = (
                    input_tensor_sequence[0].to(self.device).unsqueeze(0),
                    input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    input_tensor_sequence[2].to(self.device).unsqueeze(0),
                )

                action_tensor, logs = self.model.step(input_tensor_sequence)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)


            # create a transformation matrix for the action_robot
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R.from_euler('xyz', action_robot[3:6]).as_matrix()
            transformation_matrix[:3, 3] = action_robot[:3]

            # rotate 90 degrees about z axis
            rotation_matrix_z = R.from_euler('z', 90, degrees=True).as_matrix()
            transformation_matrix_90_z = np.eye(4)
            transformation_matrix_90_z[:3, :3] = rotation_matrix_z

            # apply the 90-degree transformation to the action_robot transformation matrix
            transformed_matrix = transformation_matrix_90_z @ transformation_matrix @ transformation_matrix_90_z.T

            # extract the updated rotation vector and translation from the transformed matrix
            updated_rotvec = R.from_matrix(transformed_matrix[:3, :3]).as_euler('xyz')
            updated_translation = transformed_matrix[:3, 3]

            # update the action_robot with the transformed values
            action_robot[:3] = updated_translation
            action_robot[3:6] = updated_rotvec



            gripper = action_tensor[-1]

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper

            logger.info("calling move_to_pose")
            self.robot.move_to_pose(action_robot[:3], action_robot[3:6], gripper)

            run_for -= 1
            self.step_n += 1

    def _set_values_by_task(self, task_name: str):
        self._opening_threshold = task_to_params_dict[task_name]["opening_threshold"]
        self._max_gripper = task_to_params_dict[task_name]["max_gripper"]
        self._gripper_threshold = task_to_params_dict[task_name]["gripper_threshold"]
        return self._max_gripper, self._gripper_threshold, self._opening_threshold

    def get_image_vanilla(self):
        while True:
            time.sleep(0.1)
            cv2_img, timestamp = self._ui_subscriber.recv_rgb_image()
            yield cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    def get_gripper_val(self):
        return self.gripper

    def _run_home(self):
        logger.info("calling home")
        self.robot.home()
        self.reset_experiment()
    def _run(self, run_for=1):
        with self._handle_interrupt():
            logger.info(f"Run robot for {run_for} step(s)")
            if self.use_depth:
                self._run_policy_depth(run_for=run_for)
            elif self.use_goals:
                self._run_policy_goals(run_for=run_for)
            else:
                self._run_policy(run_for=run_for)

    def reset_experiment(self, gripper=1.0, object_name=None):
        try:
            cv2.destroyWindow("Tracking")
        except cv2.error:
            pass
        self.async_saver.finish()
        self.run_n += 1
        self.step_n = 0
        self.gripper = gripper
        self.model.reset()
        self.image_action_buffer_manager = self.create_buffer_manager()
        self._should_home = False
        self.object_name = object_name

    def _process_instruction(self, instruction):
        global schedule
        if instruction.lower() == "h":
            # home with open gripper
            self.robot.set_home_position(lift=self.h)
            self.robot.home(gripper=1.0, reset_base=True)
            self.reset_experiment()
        elif instruction.lower() == "r":
            h = input("Enter height:")
            self.h = float(h)
            self.robot.set_home_position(lift=self.h)
        elif instruction.lower() == "hc":
            # home with closed gripper
            self.robot.set_home_position(lift=0.9, gripper=0) 
            self.robot.home(gripper=0)
            self.reset_experiment(gripper=0)
        elif instruction.lower() == "th":
            threshold_reopen = input("Enter reopening threshold: ")
            self.th = float(threshold_reopen)
            self.robot.set_home_position(lift=self.h, reopening_threshold=self.th)
        elif instruction.lower() == "mgw":
            print("This is not supported anymore")
            # max_gripper_width = input(
            #     "Enter max gripper width (155 for wide, 50 for low): "
            # )
            # self.mgw = float(max_gripper_width)
            # self.robot.set_home_position(lift=self.h, stretch_gripper_max=self.mgw)
        elif instruction.lower() == 'd':
            depth_offset = input("Enter depth offset: ")
            self.depth_offset = float(depth_offset)
        elif instruction.lower() == "gt":
            gripper_tight_value = input(
                "Enter gripper value when closed tight (default -35): "
            )
            self.gt = float(gripper_tight_value)
            self.robot.set_home_position(lift=self.h, stretch_gripper_tight=self.gt)
        elif instruction.lower() == "sg":
            sticky_gripper_value = input(
                "Enter whether gripper should be sticky (i.e. close only once, default true): "
            )
            self.sg = bool(sticky_gripper_value)
            self.robot.set_home_position(lift=self.h, sticky_gripper=self.sg)
        elif instruction.lower() == "s":
            sched_no = input("Enter schedule number:")
            base, h = schedule(int(sched_no))
            logger.info(f"h={h}, base={base}")
            self.robot.set_home_position(lift=h, base=base)
            self.schedul_no = int(sched_no)
            self.robot.home(gripper=1.0)
            self.reset_experiment()
        elif instruction.lower() == "n":
            self.schedul_no = (self.schedul_no) % 10 + 1
            base, h = schedule(self.schedul_no)
            logger.info(f"schedul_no={self.schedul_no}, h={h}, base={base}")
            self.robot.set_home_position(lift=h, base=base)
            self.robot.home(gripper=1.0)
            self.reset_experiment()
        elif instruction.lower() == "c":
            # continuous pick up, home, and place
            self.run_for = 100
            self._run(self.run_for)

            if self._should_home:
                # home with open gripper if interrupted
                self.robot.set_home_position(lift=self.h)
                self.robot.home(gripper=1.0, reset_base=True)
                self.reset_experiment()
            else:
                # home with closed gripper
                self.robot.set_home_position(lift=0.9, gripper=self.gt / self.robot.STRETCH_GRIPPER_MAX)
                self.robot.home(
                    gripper=self.gt / self.robot.STRETCH_GRIPPER_MAX
                )
                self.reset_experiment(
                    gripper=self.gt / self.robot.STRETCH_GRIPPER_MAX
                )

        elif len(instruction) == 0:
            self.run_for = 1
            self._run(self.run_for)
        elif instruction.isdigit():
            self.run_for = int(instruction)
            self._run(self.run_for)
        elif instruction.lower() == "p":
            cv2_img, np_depth, timestamp = self.subscriber.recv_image_and_depth()
            self._get_clicked_point(cv2_img, np_depth)
            point_3d = get_new_2d_point(self.object_x, self.object_y, self.object_depth, None)
            # 0.01 left, 0.04 down, 0.18 forward
            offset = np.array([-0.01, 0.04, 0.18])
            height_offset = np.array([0.0, -0.15, 0.00])
            point_3d = point_3d - offset + height_offset
            T = np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            point_3d = T @ np.append(point_3d, 1)
            translation = point_3d[:3]
            self.robot.move_to_pose(translation, np.zeros(3), 0.0)
        elif instruction.lower() == "q":
            self.async_saver.finish()
            exit()
        elif instruction.lower() == "bl":
            self.robot.move_base(0.02)
        elif instruction.lower() == "br":
            self.robot.move_base(-0.02)
        else:
            # raise warning
            logger.error("Invalid instruction")
            instruction = input("Enter instruction:")
            self._process_instruction(instruction)

    def run_continous(self):
        for i in range(20):
            logger.info(i)
            start_time = time.time()

            self.run_for = 1
            self._run(self.run_for)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)

        instruction = input("Enter instruction:")
        return instruction

    def run(self):
        time.sleep(0.5)
        self.robot.set_home_position(lift=self.h)

        while True:
            # wait for instruction
            instruction = input("Enter instruction: ")
            start_time = time.time()

            if instruction.lower() == "q":
                instruction = self._process_instruction(instruction)
                break
            elif instruction.lower() == "rc":
                self._run()
                instruction = ""
                while len(instruction) == 0:
                    instruction = self.run_continous()
                continue

            # process and send instruction to robot
            instruction = self._process_instruction(instruction)

            # continue loop only once instruction has been executed on robot

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)
            # Calculate elapsed time and sleep to maintain 10 Hz frequency
