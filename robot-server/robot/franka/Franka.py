import time
import pickle
import zmq
import numpy as np
from scipy.spatial.transform import Rotation as R
from robot.gripper import Gripper
from robot.utils import create_transform, euler_to_quat, transform_to_vec


from .deoxys_utils.messages import FrankaAction, FrankaState
from .deoxys_utils.network import create_request_socket
from .deoxys_utils.constants import CONTROL_PORT


# Home position in Franka frame (meters): 318.6mm X, 3.3mm Y, 139.4mm Z, -176.8deg Roll, -0.3deg Pitch, -48.8deg Yaw
HOME_POS = [331.3, -19.3, 288.8, -176.5, 26.3, -7.5]

# Camera-to-robot rotation (iPhone camera frame to Franka end-effector frame, including 15 deg tilt about Y)
# Camera offset in robot frame (meters): 10mm X, -15 mm Z
CAMERA_OFFSET_ROBOT = np.array([0.0, 0.0, -0.0])
CAMERA_TO_ROBOT_ROT = np.array([0, 0, 0])

class FrankaArm:
   # Default gripper values for compatibility (no hardware)
   GRIPPER_OPEN = 5400
   GRIPPER_CLOSE = 3600
   # Compatibility attributes for Stretch robot interface
   STRETCH_GRIPPER_TIGHT = [3600]  # Use GRIPPER_CLOSE as tight value
   STRETCH_GRIPPER_MAX = 5400      # Use GRIPPER_OPEN as max value
   STRETCH_GRIPPER_MIN = 3600      # Use GRIPPER_CLOSE as min value

   def __init__(self, host="localhost", port=8901):
       # 1) connect with retry mechanism
       self.req = None
       self._server_host = host
       self._server_port = port
       self._connect_with_retry(host, port)
      
       # 2) cache initial state
       self.prev_state = self.get_state()
       print("Franka initialized. EE pos:",
       self.prev_state.pos, "quat:", self.prev_state.quat)

       self.prev_pose = None  # Initialize prev_pose

       # Gripper thread is now handled internally by Gripper class
       # Loads calibration from config automatically
       self.gripper = Gripper()
       
       # Get calibrated values from loaded config
       self.GRIPPER_OPEN = self.gripper.params['range_t'][1]  # Max tick (open)
       self.GRIPPER_CLOSE = self.gripper.params['zero_t']      # Zero tick (closed)
       
       self.gripper.dxl.set_profile_acceleration(21)
       self.gripper.dxl.set_profile_velocity(125)
       self.open_gripper()

   def _connect_with_retry(self, host, port, max_retries=10, retry_delay=1.0):
       """Connect to Franka server with retry mechanism."""
       for attempt in range(max_retries):
           try:
               print(f"Attempting to connect to Franka server at {host}:{port} (attempt {attempt + 1}/{max_retries})")
               self.req = create_request_socket(host, port)
              
               # Test the connection by sending a get_state request
               if self.req is not None:
                   self.req.send(b"get_state")
                   data = self.req.recv()
                   if data != b"state_error":
                       print(f"Successfully connected to Franka server on attempt {attempt + 1}")
                       return
                   else:
                       print("Server returned state_error, retrying...")
                       self.req.close()
                       self.req = None
                  
           except Exception as e:
               print(f"Connection attempt {attempt + 1} failed: {e}")
               if self.req:
                   self.req.close()
                   self.req = None
              
           if attempt < max_retries - 1:
               print(f"Waiting {retry_delay} seconds before retry...")
               time.sleep(retry_delay)
      
       raise ConnectionError(f"Failed to connect to Franka server at {host}:{port} after {max_retries} attempts")


   def get_state(self):
       """Fetch the latest EE position/quaternion."""
       if self.req is None:
           raise ConnectionError("No connection to Franka server")
       try:
           # Ensure ZMQ REQ sockets is not in the middle of a request-reply cycle
           # If the socket is in an invalid state, recreate it
           try:
               self.req.send(b"get_state", flags=0)
           except zmq.error.ZMQError as e:
               if "Operation cannot be accomplished in current state" in str(e):
                   print("ZMQ socket in invalid state, recreating connection...")
                   self.req.close()
                   self.req = create_request_socket(self._server_host, self._server_port)
                   self.req.send(b"get_state", flags=0)
               else:
                   raise
          
           data = self.req.recv()
           return pickle.loads(data)
       except Exception as e:
           print(f"Error getting state: {e}")
           raise
  
   def get_position(self):
       """Return [x, y, z, roll, pitch, yaw] in world frame.
       Built to replicate xArm/RUM inputs as Franka returns quaternions
       """
       state = self.get_state()

       pos = np.array(state.pos)
       quat = np.array(state.quat)

       r, p, y = R.from_quat(quat).as_euler("xyz", degrees=True)

       return np.concatenate([pos, [r,p,y]], axis=0)

   def open_gripper(self):
       """Fully open the gripper."""
       self.gripper.set_desired_position(1.0)
  
   def close_gripper(self):
       """Fully close the gripper."""
       self.gripper.set_desired_position(0.0)

   def set_home_position(self, lift=None, base=None, stretch_gripper_max=None, closing_threshold=None, reopening_threshold=None, stretch_gripper_tight=None, gripper=None):
       """
       Created for controller compatability from Stretch Controller.
       Currently doesn't do anything of importance for Franka run.
       """
       # Update compatibility values if provided
       if stretch_gripper_max is not None:
           self.STRETCH_GRIPPER_MAX = stretch_gripper_max
       if stretch_gripper_tight is not None:
           self.STRETCH_GRIPPER_TIGHT = [stretch_gripper_tight] if not isinstance(stretch_gripper_tight, list) else stretch_gripper_tight
       # Note: gripper parameter ignored for Franka compatibility

   def home(self, gripper=1.0, reset_base=False):
       """Reset the arm to its default home position."""
       # Using reset=True will invoke the Franka server's reset logic
       # Note: reset_base parameter ignored for Franka compatibility
       action = FrankaAction(
           pos       = np.array([0.0, 0.0, 0.0]),
           quat      = np.array([0.0, 0.0, 0.0, 1.0]),
           gripper   = np.array([1.0]),
           reset     = True,
           timestamp = time.time(),
       )
       if self.req is None:
           raise ConnectionError("No connection to Franka server")
      
       try:
           self.req.send(pickle.dumps(action, protocol=-1), flags=0)
       except zmq.error.ZMQError as e:
           if "Operation cannot be accomplished in current state" in str(e):
               print("ZMQ socket in invalid state during home, recreating connection...")
               self.req.close()
               self.req = create_request_socket(self._server_host, self._server_port)
               self.req.send(pickle.dumps(action, protocol=-1), flags=0)
           else:
               raise
      
       self.prev_state = self.get_state()
       print("Homed. EE pos:",
             self.prev_state.pos, "quat:", self.prev_state.quat)
       self.gripper.set_desired_position(1.0)

   def move_to_pose(self, translation: np.ndarray, rotation_euler: np.ndarray, gripper: float, use_prev: bool = False, prev_pose: bool = False) -> None:
       """
       translation: (3,) meters [dx,dy,dz] in iPhone camera frame
       rotation_euler: (3,) radians [droll,dpitch,dyaw] in iPhone camera frame
       gripper: GRIPPER_OPEN or GRIPPER_CLOSE
       use_prev: if True, chain off last commanded pose; else use live state
       prev_pose: (optional, for interface compatibility) if provided, overrides use_prev
       """
       if prev_pose is not None:
           use_prev = prev_pose
      
       start_time = time.time()
       # Translations
       # Input X -> -Z
       # Input Y -> X
       # Input Z -> Y
       #
       # Rotations
       # Input Roll -> -Yaw
       # Input Pitch -> -Roll
       # Input Yaw -> Pitch
       Camera_Offset = np.concatenate([CAMERA_OFFSET_ROBOT, CAMERA_TO_ROBOT_ROT])
       Camera_Offset_Matrix = create_transform(Camera_Offset)
       Franka_To_Camera = np.array([
           [0,0,-1,0],
           [0,1,0,0],
           [1,0,0,0],
           [0,0,0,1],
       ])
      
       input_vec = np.concatenate([translation*0.8, np.degrees(rotation_euler)]) 
       input_transform = create_transform(input_vec)

       # 1) Get current state
       current_state = self.get_state()
       current_pos = np.array(current_state.pos)  # meters
       current_quat = np.array(current_state.quat)  # quaternion
       current_euler = R.from_quat(current_quat).as_euler('xyz', degrees=True)
    #    print(f"Using current live state: pos={current_pos}, euler={current_euler}")

       # 2) Apply the delta action in robot frame
       current_transform = np.eye(4)
       current_transform[:3, :3] = R.from_quat(current_quat).as_matrix()
       current_transform[:3, 3] = current_pos

    #    print(f"Current transform: {current_transform}")
    #    print(f"Input transform: {input_transform}")
    #    print(f"Franka to Camera: {Franka_To_Camera}")
    #    print(f"Camera offset matrix: {Camera_Offset_Matrix}")

       new_transform = current_transform @ Camera_Offset_Matrix @ Franka_To_Camera @ input_transform @ Franka_To_Camera.T @ np.linalg.inv(Camera_Offset_Matrix)
    #    print(f"New transform: {new_transform}")

       new_vec = transform_to_vec(new_transform)
       new_pos = new_vec[:3]
       new_euler = new_vec[3:]
       new_quat = R.from_euler('xyz', new_euler, degrees=True).as_quat()

       # 3) Send action to Franka
       action = FrankaAction(
           pos=np.array(new_pos),
           quat=np.array(new_quat),
           gripper=np.array([float(gripper)]),
           reset=False,
           timestamp=time.time(),
       )

       if self.req is None:
           raise ConnectionError("No connection to Franka server")

       # REQ/REP pattern: send exactly once, then receive exactly once
       try:
           self.req.send(pickle.dumps(action, protocol=-1), flags=0)
       except zmq.error.ZMQError as e:
           if "Operation cannot be accomplished in current state" in str(e):
               print("ZMQ socket in invalid state during move, recreating connection...")
               self.req.close()
               self.req = create_request_socket(self._server_host, self._server_port)
               self.req.send(pickle.dumps(action, protocol=-1), flags=0)
           else:
               raise

       # Receive the server's state reply for this action and update caches
       try:
           data = self.req.recv()
           if data != b"state_error":
               self.prev_state = pickle.loads(data)
               # Convert prev_state to pose for logging and chaining
               pos = np.array(self.prev_state.pos)
               quat = np.array(self.prev_state.quat)
               eul = R.from_quat(quat).as_euler('xyz', degrees=True)
               self.prev_pose = np.concatenate([pos, eul], axis=0)
            #    print(f"New pose: {self.prev_pose}")
           else:
               print("Warning: server returned state_error after action")
       except Exception as e:
           print(f"Error receiving state after move: {e}")

       print(f"Time taken to move to pose: {time.time() - start_time} seconds")

        self.gripper.set_desired_position(gripper)

   def remember_pose(self):
       """Store the current end-effector pose for later use."""
       # Use cached state to avoid redundant ZMQ request
       if self.prev_state is not None:
           pos = np.array(self.prev_state.pos)
           quat = np.array(self.prev_state.quat)
           eul = R.from_quat(quat).as_euler('xyz', degrees=True)
           self.prev_pose = np.concatenate([pos, eul], axis=0)
       else:
           # Fallback to fresh request if no cached state
           self.prev_pose = self.get_position()
       print(f"Remembered pose: {self.prev_pose}")   

