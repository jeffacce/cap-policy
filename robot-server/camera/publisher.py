from camera.demo import R3DApp
import cv2
import time
from robot.zmq_utils import *
import os
from scipy.spatial.transform import Rotation as R
    
class R3DCameraPublisher(ProcessInstantiator):
    def __init__(self, host, camera_port, pose_port, stream_depth, stream_pose):
        super().__init__()
        self.host = host
        self.camera_port = camera_port
        self.pose_port = pose_port
        self.stream_depth = stream_depth
        self.stream_pose = stream_pose
        self.rgb_publisher = ZMQCameraPublisher(
            host = self.host, 
            port = self.camera_port
        )
        if self.stream_pose:
            self.pose_publisher = ZMQKeypointPublisher(
                host = self.host, 
                port = self.pose_port
            )
        
        self._seq = 0
        self.timer = FrequencyTimer(30)

        self._start_camera()

    # start the Record3D streaming
    def _start_camera(self):
        self.app = R3DApp()
        while self.app.stream_stopped:
            try:
                self.app.connect_to_device(dev_idx=0)
            except RuntimeError as e:
                print(e)
                print(
                    "Retrying to connect to device with id {idx}, make sure the device is connected and id is correct...".format(
                        idx=0
                    )
                )
                time.sleep(2)

    # get the RGB and depth images from the Record3D
    def get_rgb_depth_images(self):
        image = None
        while image is None:
            image, depth, pose = self.app.start_process_image()
            image = np.moveaxis(image, [0], [1])[..., ::-1, ::-1]
            image = np.rot90(image, 2)
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        if self.stream_depth: 
            depth = np.ascontiguousarray(np.rot90(depth, 1)).astype(np.float64)
            return image, depth, pose
        else:
            return image, pose
    
    # get RGB images at 50Hz and publish them to the ZMQ port
    def stream(self):
        while True:
            if self.app.stream_stopped:
                try:
                    self.app.connect_to_device(dev_idx=0)
                except RuntimeError as e:
                    print(e)
                    print(
                        "Retrying to connect to device with id {idx}, make sure the device is connected and id is correct...".format(
                            idx=0
                        )
                    )
                    time.sleep(2)
            else:
                self.timer.start_loop()
                if self.stream_depth:
                    wrist_image, wrist_depth, pose = self.get_rgb_depth_images()
                    self.rgb_publisher.pub_image_and_depth(wrist_image, wrist_depth, time.time())
                else:
                    wrist_image, pose = self.get_rgb_depth_images()
                    self.rgb_publisher.pub_rgb_image(wrist_image, time.time())   

                if self.stream_pose:
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = R.from_quat(pose[:4]).as_matrix()
                    transformation_matrix[:3, 3] = pose[4:]

                    # rotate 90 degrees about z axis
                    rotation_matrix_z = R.from_euler('z', 90, degrees=True).as_matrix()
                    transformation_matrix_90_z = np.eye(4)
                    transformation_matrix_90_z[:3, :3] = rotation_matrix_z

                    # apply the 90-degree transformation to the action_robot transformation matrix
                    transformed_matrix = transformation_matrix_90_z @ transformation_matrix @ transformation_matrix_90_z.T

                    # extract the updated rotation vector
                    updated_rotvec = R.from_matrix(transformed_matrix[:3, :3]).as_euler('xyz', degrees=False)
                    updated_rotvec[2] -= np.pi / 2
                    updated_rotvec = R.from_euler('xyz', updated_rotvec, degrees=False).as_quat()

                    pose = np.concatenate([updated_rotvec, pose[4:]])
                    self.pose_publisher.pub_keypoints(pose, "pose")

                self.timer.end_loop()

                if "DISPLAY" in os.environ:
                    cv2.imshow("iPhone", wrist_image)
                    if self.stream_depth:
                        cv2.imshow("Depth", wrist_depth)
            
                if cv2.waitKey(1) == 27:
                    break
        
        cv2.destroyAllWindows()