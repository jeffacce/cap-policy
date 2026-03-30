from xarm.wrapper import XArmAPI
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R, Slerp
from robot.utils import create_transform, transform_to_vec
from robot.gripper import Gripper

HOME_POS = [180, -1.13, 280.0, 180, 20, 0]
#HOME_POS = [221.008911, -1.13, 95.083954, 180, 0, 0]

END_EFFECTOR_TO_IPHONE = [60,0,-60,0,75,0]

# xArm coordinate system:
"""
       x      y
       |    /
       |  /
z ______|/
"""

class xArm:
   def __init__(self, xarm_ip):
       '''
       This function is used to initialize the xArm:
       xarm_ip: the IP address of the xArm
       '''
       self.arm = XArmAPI(xarm_ip)
       self.arm.connect()
       self.arm.motion_enable(enable=True)
       self.arm.set_mode(1)
       self.arm.set_state(0)
      
       print('xArm initialized')
      
       self.wrist_to_iphone = create_transform(END_EFFECTOR_TO_IPHONE)
       self.base_to_home = create_transform(HOME_POS)

       # Initialize gripper (loads calibration from config automatically)
       self.gripper = Gripper()
       
       # Get calibrated values from loaded config
       self.GRIPPER_OPEN = self.gripper.params['range_t'][1]  # Max tick (open)
       self.GRIPPER_CLOSE = self.gripper.params['zero_t']      # Zero tick (closed)

       # Update compatibility values with calibrated values
       self.STRETCH_GRIPPER_MAX = self.GRIPPER_OPEN      # Use calibrated open as max
       self.STRETCH_GRIPPER_MIN = self.GRIPPER_CLOSE     # Use calibrated close as min 
       self.STRETCH_GRIPPER_TIGHT = [self.GRIPPER_CLOSE] # Use calibrated close as tight

       print(f'Loaded gripper values from config: OPEN={self.GRIPPER_OPEN}, CLOSE={self.GRIPPER_CLOSE}')
      
       # Gripper thread is now handled internally by Gripper class
       self.gripper.dxl.set_profile_acceleration(20)
       self.gripper.dxl.set_profile_velocity(300)
       self.open_gripper()
       self.prev_pose = None

       # Movement Thread Components
       self._cmd_lock = threading.Lock()
       self._latest_cmd = None
       self._interrupt = threading.Event()
       self._ctrl_lock = threading.Lock()
       self._ctrl_cmd = None
      
       # Start movement thread
       self._movement_thread = threading.Thread(target=self._movement_loop, daemon=True)
       self._movement_thread.start()

       print("XArm initialized: ✓✓✓")

   def switch_to_servo_mode(self):
       """Safely switch to servo mode"""
       # Wait for any current motion to complete
       while self.arm.get_is_moving():
           time.sleep(0.1)
       time.sleep(0.1)
       # Ensure robot is enabled and ready
       self.arm.motion_enable(enable=True)
       self.arm.set_state(0)
      
       # Switch to servo mode
       self.arm.set_mode(1)  # Servo mode
       self.arm.set_state(0)  # Ready state

   def switch_to_position_mode(self):
       """Safely switch to position mode"""
       # Wait for any current motion to complete
       while self.arm.get_is_moving():
           time.sleep(0.1)
      
       time.sleep(0.1)
      
       # Ensure robot is enabled and ready
       self.arm.motion_enable(enable=True)
       self.arm.set_state(0)
      
       # Switch to position mode
       self.arm.set_mode(0)  # Position mode
       self.arm.set_state(0)  # Ready state

   def set_home_position(self, lift=None, base=None, stretch_gripper_max=None, closing_threshold=None, reopening_threshold=None, stretch_gripper_tight=None, gripper=None):
       # Update compatibility values if provided
       if stretch_gripper_max is not None:
           self.STRETCH_GRIPPER_MAX = stretch_gripper_max
       if stretch_gripper_tight is not None:
           self.STRETCH_GRIPPER_TIGHT = [stretch_gripper_tight] if not isinstance(stretch_gripper_tight, list) else stretch_gripper_tight

   def home(self, gripper=1.0, reset_base=False):
       '''
       This function is used to move the robot to the home position:
       gripper: gripper position - 0-1
       reset_base: whether to reset the base - bool
       '''
       self.home_interpolation(v_mm_s=150.0, w_deg_s=45.0, dt=0.01)

   # moves from home position to run starting position given relative motion from server
   def move_relative(self, relative_action):
       '''
       This function is used to move the robot relative to the current pose:
       relative_action: relative action vector - m
       '''
    #    print(relative_action)
       relative_action = np.array(relative_action)[:-1] # convert to numpy array get rid of gripper value
       relative_action[:3] = 1000 * np.array([[1,0,0],[0,0,-1],[0,1,0]]) @ relative_action[:3] # swap y and -z, and convert to mm

       new_pos = transform_to_vec(self.base_to_home @ create_transform(relative_action))
          
       self.arm.set_position(*new_pos, speed=50, mvacc=1000, wait=False)
  
   def remember_pose(self):
       '''
       This function is used to remember the current pose of the robot:
       '''
       code, self.prev_pose = self.arm.get_position(is_radian=False)
       return code

   def move_to_pose(self, translation, rotation, gripper, prev_pose=False):
       '''
       This function is used to receive movement commands from the policy:
       translation: translation vector - m
       rotation: rotation vector - rad
       gripper: gripper position - 0-1
       prev_pose: whether to use the previous pose - bool
       '''
       with self._cmd_lock:
           self._latest_cmd = (translation, rotation, gripper, prev_pose)
           self._interrupt.set()
          
       self.gripper.set_desired_position(gripper)
       return {"status": "accepted"} 

   def close_gripper(self):
       '''
       This function is used to close the gripper:
       '''
       self.gripper.set_desired_position(0.0)

   def open_gripper(self):
       '''
       This function is used to open the gripper:
       '''
       self.gripper.set_desired_position(1.0)

   def move_to_pose_interpolation(self, translation, rotation, gripper, prev_pose=False):
       '''
       This function is used to move the robot to a target pose:
       translation: translation vector - m
       rotation: rotation vector - rad
       gripper: gripper position - 0-1
       prev_pose: whether to use the previous pose - bool
       '''
       # Build the target pose (camera Δ -> base pose)
       relative_action = np.concatenate([translation, rotation])  # [x,y,z,Rx,Ry,Rz], m & rad in camera frame
       code, current_pos = self.arm.get_position(is_radian=False)
       if code != 0:
           return

       # ---- camera-frame to base-frame  ----
       relative_action[:3] *= 1000.0      # m -> mm
       relative_action[0]  *= -1.0        # flip X
       relative_action[2]  *= -1.0        # flip Z
       relative_action[3:]  = np.rad2deg(relative_action[3:])  # rad -> deg
       relative_action[3]  *= -1.0        # flip Rx
       relative_action[5]  *= -1.0        # flip Rz

       T_base_tcp   = create_transform(current_pos)
       T_cam_delta  = create_transform(relative_action)
       T_tcp_cam    = self.wrist_to_iphone
       T_target     = T_base_tcp @ T_tcp_cam @ T_cam_delta @ np.linalg.inv(T_tcp_cam)

       start_pose   = np.array(transform_to_vec(T_base_tcp))     # [x,y,z,Rx,Ry,Rz]  (mm, deg)
       target_pose  = np.array(transform_to_vec(T_target))       # [x,y,z,Rx,Ry,Rz]  (mm, deg)

       # ---- plan a synchronized duration ----
       v_mm_s = 150.0    # linear speed cap - mm/s
       w_deg_s = 45.0    # angular speed cap - deg/s
       dt = 0.01         # time step - s

       delta_pos = target_pose[:3] - start_pose[:3]
       dist_mm   = float(np.linalg.norm(delta_pos))

       # shortest rotation distance via quaternions
       R0 = R.from_euler('xyz', start_pose[3:], degrees=True)
       R1 = R.from_euler('xyz', target_pose[3:], degrees=True)

       # Approximate angular distance (geodesic) in degrees:
       rel = R1 * R0.inv()           # or R0.inv() * R1 — same angle
       ang_rad = rel.magnitude()     # shortest geodesic angle (radians)
       ang_deg = np.degrees(ang_rad)

       # total duration so both finish together
       T_lin = dist_mm / v_mm_s if v_mm_s > 0 else 0.0
       T_rot = ang_deg / w_deg_s if w_deg_s > 0 else 0.0
       T = max(T_lin, T_rot, 1e-6)  # avoid zero

       # ---- set up SLERP ----
       slerp = Slerp([0.0, 1.0], R.from_euler('xyz', [start_pose[3:], target_pose[3:]], degrees=True))
  
       t_elapsed = 0.0
       while True:
           if self._interrupt.is_set():
               return

           f = min(1.0, t_elapsed / T)
           # LERP translation
           pos_f = start_pose[:3] + f * delta_pos

           # SLERP result
           rot_f = slerp([f])[0].as_euler('xyz', degrees=True)

           # Build one 6-element pose vector
           mvpose = np.concatenate([pos_f, rot_f]).astype(float).tolist()

           ret = self.arm.set_servo_cartesian(mvpose, speed=0.1, mvacc=0.1)
           if f >= 1.0:
               break
           time.sleep(dt)
           t_elapsed += dt

   def home_interpolation(self, v_mm_s=150.0, w_deg_s=45.0, dt=0.01):
       '''
       This function is used to move the robot to the home position:
       v_mm_s: linear speed cap - mm/s
       w_deg_s: angular speed cap - deg/s
       dt: time step - s
       '''
       # open gripper first
       self.open_gripper()

       # read current base-frame pose (mm, deg)
       code, cur = self.arm.get_position(is_radian=False)
       if code != 0:
           return
       start_pose  = np.array(cur, dtype=float)         # [x,y,z,Rx,Ry,Rz] (mm,deg)
       target_pose = np.array(HOME_POS, dtype=float)    # same units/frame

       # plan synchronized duration
       delta_pos = target_pose[:3] - start_pose[:3]
       dist_mm   = float(np.linalg.norm(delta_pos))
       R0 = R.from_euler('xyz', start_pose[3:],  degrees=True)
       R1 = R.from_euler('xyz', target_pose[3:], degrees=True)

       # shortest angular distance (radians)
       rel = R1 * R0.inv()
       ang_rad = getattr(rel, "magnitude", None)() if hasattr(rel, "magnitude") else np.linalg.norm(rel.as_rotvec())
       ang_deg = np.degrees(ang_rad)

       T_lin = dist_mm / v_mm_s if v_mm_s > 0 else 0.0
       T_rot = ang_deg / w_deg_s if w_deg_s > 0 else 0.0
       T     = max(T_lin, T_rot, 1e-6)

       # SLERP over [0,1]
       slerp = Slerp([0.0, 1.0],
                   R.from_euler('xyz', np.vstack([start_pose[3:], target_pose[3:]]), degrees=True))

       t_elapsed = 0.0
       while True:
           if self._interrupt.is_set():
               return
           f = min(1.0, t_elapsed / T)
           # LERP translation
           pos_f = start_pose[:3] + f * delta_pos

           # SLERP result
           rot_f = slerp([f])[0].as_euler('xyz', degrees=True)

           # Build one 6-element pose vector
           mvpose = np.concatenate([pos_f, rot_f]).astype(float).tolist()

           ret = self.arm.set_servo_cartesian(mvpose, speed=0.1, mvacc=0.1)
           if f >= 1.0:
               break
           time.sleep(dt)
           t_elapsed += dt

   def _movement_loop(self):
       '''
       This function is used as the listener for the movement thread.
       '''
       while True:
           self._interrupt.wait()
           # 1) Handle control commands
           with self._ctrl_lock:
               ctrl = self._ctrl_cmd
               self._ctrl_cmd = None
           if ctrl:
               kind, done, kwargs = ctrl
               if kind == 'home':
                   self._execute_home(done)
                   # continue to next loop (don’t run motion in same cycle)
                   continue

           # 2) Otherwise, run the latest motion command (if any)
           with self._cmd_lock:
               cmd = self._latest_cmd
               self._latest_cmd = None
               # clear interrupt so we can detect new arrivals mid-trajectory
               self._interrupt.clear()
           if cmd:
               self.move_to_pose_interpolation(*cmd)

