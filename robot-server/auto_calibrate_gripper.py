import cv2
import time
import numpy as np
from multiprocessing import Event
from record3d import Record3DStream
from robot.gripper import Gripper


class R3DApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.stream_stopped = True

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        self.stream_stopped = True
        print("Stream stopped")

    def connect_to_device(self, dev_idx):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        print("{} device(s) found".format(len(devs)))
        for dev in devs:
            print("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(dev_idx)
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing
        self.stream_stopped = False

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def start_process_image(self):
        self.event.wait(5)
        rgb = self.session.get_rgb_frame()
        depth = self.session.get_depth_frame()
        camera_pose = self.session.get_camera_pose()
        pose = np.array(
            [
                camera_pose.qx,
                camera_pose.qy,
                camera_pose.qz,
                camera_pose.qw,
                camera_pose.tx,
                camera_pose.ty,
                camera_pose.tz,
            ]
        )
        return rgb, depth, pose


class R3DCamera:
    def __init__(self):
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

    def get_rgb(self):
        image = None
        while image is None:
            image, depth, pose = self.app.start_process_image()
            image = np.moveaxis(image, [0], [1])[..., ::-1, ::-1]
            image = np.rot90(image, 2)
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            return image


def get_gripper_width_from_vision(camera):
    """
    Fetches an image from the camera, processes it, and returns
    the pixel distance between the two blue markers.
    """
    frame = camera.get_rgb()
    if frame is None:
        print("Warning: Received an empty image frame.")
        return None
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_blue = np.array([95, 70, 80]) # [H_min, L_min, S_min]
    upper_blue = np.array([125, 200, 255]) # [H_max, L_max, S_max]
    
    mask = cv2.inRange(hls_frame, lower_blue, upper_blue)
    N = len(mask) // 2
    left = mask[:, :N]
    right = mask[:, N:]

    left_center = np.where(left)[1].mean()
    right_center = np.where(right)[1].mean() + N
    width = right_center - left_center
    return width


def move_by_until(g, camera, move_by, sleep=1.0, until="stalled", thresh=0.5):
    assert until in ["stalled", "moved"]
    width = get_gripper_width_from_vision(camera)
    ticks = None
    while True:
        g.move_by(move_by)
        time.sleep(sleep)
        g.pull_status()
        new_width = get_gripper_width_from_vision(camera)
        new_ticks = g.status['pos_ticks']
        if until == "stalled":
            if abs(new_width - width) < thresh:
                return width, ticks
        elif until == "moved":
            if abs(new_width - width) >= thresh:
                return new_width, new_ticks
        width = new_width
        ticks = new_ticks

def calibrate_gripper(g=None):
        '''
        Calibrates the gripper using the vision system.
        Args:
            g: The gripper object.
        Returns:
            None
        '''
        # == GRIPPER == 
        print("Starting gripper...")
        if g is None:
            g = Gripper()
        if not g.is_ready():
            print("Gripper startup failed. Exiting.")
            return
        print("Gripper startup successful.")

        g.params['zero_t'] = g.dxl.zero() 
        zero_t = g.params['zero_t']
        g.params['range_t'] = [max(0, zero_t - 50000), zero_t + 50000]  # Wide range around zero
        
        if g.params['flip_encoder_polarity']:
            wr_max = g.ticks_to_world_rad(g.params['range_t'][0])
            wr_min = g.ticks_to_world_rad(g.params['range_t'][1])
        else:
            wr_max = g.ticks_to_world_rad(g.params['range_t'][1])
            wr_min = g.ticks_to_world_rad(g.params['range_t'][0])
        g.soft_motion_limits = {'collision': [None, None], 'user': [None, None], 'hard': [wr_min, wr_max],
                                'current': [wr_min, wr_max]}

        # == 2 CAMERA == 
        print("Initializing camera")
        camera = R3DCamera()
        print("Camera created. Waiting for first frame...")
        rgb = camera.get_rgb()
        if rgb is None:
            raise Exception("Failed to init")
        print("Camera initialized.")

        print("Finding open position using vision...")
        move_by_until(g, camera, move_by=-500, sleep=1.0, until="stalled")  # fully opened (negative = open)
        
        open_width, open_ticks = move_by_until(g, camera, move_by=100, sleep=1.0, until="moved")  # backtrack: close slightly to detect movement
        print(f"Done opening at {open_width} pixels, {open_ticks} ticks")

        # Update range with global ticks: [zero_t (closed), open_ticks (open)]
        g.params['range_t'] = [g.params['zero_t'], open_ticks]

        # Calculate percentage value for reporting
        stretch_gripper_max_value = g.world_rad_to_pct(g.ticks_to_world_rad(g.params['range_t'][1]))

        # Save to config
        g.write_configuration_param_to_YAML('stretch_gripper.range_t', g.params['range_t'])
        g.write_configuration_param_to_YAML('stretch_gripper.zero_t', g.params['zero_t'])
        
        print('---------------------------------------------------')
        print("\nCALIBRATION COMPLETE\n")
        print(f"  Zero Ticks: {g.params['zero_t']}")
        print(f"  Open Ticks: {open_ticks}")
        print(f"  Range: [{g.params['range_t'][0]}, {g.params['range_t'][1]}]")
        print(f"  Calculated Percentage Value: {stretch_gripper_max_value}\n")
        print(f"\n{stretch_gripper_max_value}\n")
        print('---------------------------------------------------')

        g.stop()
        print("Calibration script finished.")



def main():
    print("\n" + "="*50)
    print("Please remove blue objects from the camera view..")
    input("Press Enter when ready to continue with calibration...")


    try: 
        import stretch_body.stretch_gripper as gripper
        is_Stretch = True
    except:
        is_Stretch = False
    
    if is_Stretch:
        print("Starting Stretch gripper...")
        g = gripper.StretchGripper()
        if not g.startup():
            print("Gripper startup failed. Exiting.")
            return
        print("Gripper startup successful.")

        g.params['zero_t'] = 4000
        g.params['range_t'] = [0, 100000]
        
        if g.params['flip_encoder_polarity']:
            wr_max = g.ticks_to_world_rad(g.params['range_t'][0])
            wr_min = g.ticks_to_world_rad(g.params['range_t'][1])
        else:
            wr_max = g.ticks_to_world_rad(g.params['range_t'][1])
            wr_min = g.ticks_to_world_rad(g.params['range_t'][0])
        g.soft_motion_limits = {'collision': [None, None], 'user': [None, None], 'hard': [wr_min, wr_max],
                                   'current': [wr_min, wr_max]}

        # == 2 CAMERA == 
        print("Initializing camera")
        camera = R3DCamera()
        print("Camera created. Waiting for first frame...")
        rgb = camera.get_rgb()
        if rgb is None:
            raise Exception("Failed to init")
        print("Camera initialized.")

        # == 3 HOMING ==
        print(f"Homing twice...")
        for i in range(2):
            g.home()
            time.sleep(0.5)

        g.move_to(0.0)
        time.sleep(2.0) # Wait for move to complete
        
        move_by_until(g, camera, move_by=-5, sleep=1.0, until="moved")  # ensure it starts closing
        move_by_until(g, camera, move_by=-10, sleep=1.0, until="stalled")  # fully closed
        close_width, close_ticks = move_by_until(g, camera, move_by=5, sleep=1.0, until="moved")  # ensure it starts opening
        print(f"Done closing at {close_width}, {close_ticks}")
        
        move_by_until(g, camera, move_by=10, sleep=1.0, until="stalled")  # fully opened
        open_width, open_ticks = move_by_until(g, camera, move_by=-5, sleep=1.0, until="moved")  # backtrack to see slight closing movement
        print(f"Done opening at {open_width}, {open_ticks}")

        g.params['zero_t'] = close_ticks
        g.params['range_t'] = [0, open_ticks]

        stretch_gripper_max_value = g.world_rad_to_pct(g.ticks_to_world_rad(g.params['range_t'][1]))

        g.write_configuration_param_to_YAML('stretch_gripper.range_t', g.params['range_t'])
        g.write_configuration_param_to_YAML('stretch_gripper.zero_t', g.params['zero_t'])
        
        print('---------------------------------------------------')
        print("\nCALIBRATION COMPLETE\n")
        print(f"  Calibrated Max Ticks: {open_ticks}")
        print(f"  Calculated Percentage Value: {stretch_gripper_max_value}\n")
        print(f"\n{stretch_gripper_max_value}\n")
        print('---------------------------------------------------')

        g.stop()
        print("Calibration script finished.")
    else:
        print("Starting Dynamixel gripper...")
        g = Gripper()
        if not g.is_ready():
            print("Gripper startup failed. Exiting.")
            return
        print("Gripper startup successful.")
        
        # Run calibration with new flow (hardware zero + vision open)
        calibrate_gripper(g)


if __name__ == '__main__':
    main()
