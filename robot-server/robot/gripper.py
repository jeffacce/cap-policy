
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
from robot.dxl import DXL
import os
import glob
import time
import threading
import yaml

MY_DXL = 'X_SERIES'
BAUDRATE = 115200

# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION = 2.0

# Factory default ID of all DYNAMIXEL is 1
DXL_ID = 1

# Use the actual port assigned to the U2D2.
DEFAULT_DEVICENAME = '/dev/ttyUSB0'

# Common USB device names to try
COMMON_DEVICES = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyUSB4', '/dev/ttyUSB5']

# Comprehensive scanning parameters
COMMON_PROTOCOLS = [2.0, 1.0]
COMMON_BAUDS = [57600, 115200] # Alternate Baudrates: 1000000, 2000000, 3000000, 4000000, 4500000
ID_CANDIDATES = [1, 0] + list(range(2, 21))  # try 1 first, then 0, then 2..20

MIN_TOTAL_LOOPS = 15


def find_dynamixel_device():
    """
    Comprehensive auto-detection of Dynamixel device by scanning all USB devices
    with multiple protocols, baud rates, and IDs.
    Returns a dictionary with device configuration or None if none found.
    """
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

    # Get all available USB devices
    devices = (sorted(glob.glob("/dev/ttyUSB*")) + 
            sorted(glob.glob("/dev/ttyACM*")))
    
    if not devices:
        print("No USB devices found")
        return None

    print("Auto-detecting Dynamixel gripper device...")
    print(f"Scanning {len(devices)} devices: {', '.join(devices)}")

    for dev in devices:
        if not os.path.exists(dev):
            print(f"  ✗ {dev} (missing)")
            continue

        for proto in COMMON_PROTOCOLS:
            port = PortHandler(dev)
            pack = PacketHandler(proto)

            # Try each baud on this (dev, proto)
            for baud in COMMON_BAUDS:
                try:
                    if not port.openPort():
                        continue
                    if not port.setBaudRate(baud):
                        port.closePort()
                        continue

                    # Quick small pause lets adapters settle
                    time.sleep(0.01)

                    # Try likely IDs; break on first success
                    for dxl_id in ID_CANDIDATES:
                        # PING: returns (model_number, comm_result, dxl_error) in recent SDKs
                        try:
                            model, comm_res, dxl_err = pack.ping(port, dxl_id)
                        except TypeError:
                            # Older SDK signature: model, comm_res = pack.ping(port, dxl_id)
                            model, comm_res = pack.ping(port, dxl_id)
                            dxl_err = 0

                        if comm_res == COMM_SUCCESS and dxl_err == 0 and model not in (None, 0):
                            print(f"  ✓ Found Dynamixel @ {dev} | proto {proto} | {baud} bps | ID {dxl_id} | model {model}")
                            port.closePort()
                            return {
                                'device_name': dev,
                                'protocol_version': proto,
                                'baudrate': baud,
                                'dxl_id': dxl_id,
                                'model': model
                            }

                    # No ID at this baud—close before trying next baud
                    port.closePort()

                except Exception as e:
                    # Ensure the port is closed on errors
                    try: 
                        port.closePort()
                    except: 
                        pass
                    print(f"  ! {dev} proto {proto} baud {baud}: {e}")
                    # Continue scanning other combos

    print("  ✗ No Dynamixel device found on any ports")
    return None


class Gripper:
    def __init__(self, dxl_id=None, baudrate=None):
        if dxl_id is None and baudrate is None:
            detected_config = find_dynamixel_device()
            if detected_config is None:
                print(f"Warning: No Dynamixel device found, using default: {DEFAULT_DEVICENAME}")
                self.devicename = DEFAULT_DEVICENAME
                use_dxl_id = DXL_ID
                use_baudrate = BAUDRATE
                self.dxl = DXL(self.devicename, PROTOCOL_VERSION, use_baudrate, use_dxl_id)
            else:
                print(f"Using auto-detected device configuration: {detected_config['device_name']}")
                self.devicename = detected_config['device_name']
                use_dxl_id = detected_config['dxl_id']
                use_baudrate = detected_config['baudrate']
                self.dxl = DXL(
                    detected_config['device_name'],
                    detected_config['protocol_version'],
                    use_baudrate,
                    use_dxl_id
                )
            
            self.dxl.set_return_delay_time(0)
            self.dxl.enable_torque()
            self.dxl.set_pos_d_gain(0)
            self.dxl.set_profile_acceleration(21)
            self.dxl.set_profile_velocity(125)
        else:
            self.dxl = None
            self.devicename = None
        
        # Initialize parameters dictionary
        self.params = {
            'zero_t': 0,              # Zero position in ticks (absolute)
            'range_t': [0, 100000],     # Range [min, max] in ticks (absolute)
            'flip_encoder_polarity': False  # Whether to flip encoder direction
        }
        self.status = {'pos_ticks': 100000}
        self.counter = 0

        # Load parameters from config if available
        self._load_params_from_config()
        
        # Gripper thread components (moved from xArm/Franka)
        self._gripper_lock = threading.Lock()
        self._gripper_cmd = None
        self._gripper_interrupt = threading.Event()
        self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_thread.start()
    
    def _normalized_to_ticks(self, normalized_value):
        """
        Convert normalized (0-1) value to absolute ticks.
        0 = closed (zero_t), 1 = open (range_t[1])
        Args:
            normalized_value: Value between 0.0 (closed) and 1.0 (open)
        Returns:
            Absolute position in ticks
        """
        zero_t = self.params['zero_t']  # Closed position
        open_tick = self.params['range_t'][1]  # Open position
        # Clamp normalized value to [0, 1]
        normalized_value = max(0.0, min(1.0, normalized_value))
        # Convert: zero_t + (open_tick - zero_t) * normalized_value
        # This works whether open_tick > zero_t or open_tick < zero_t
        target_ticks = zero_t + (open_tick - zero_t) * normalized_value
        return int(target_ticks)
    
    def set_desired_position(self, position, normalized=True):
        """
        Non-blocking command to move gripper to position.
        Args:
            position: Target position value
            normalized: If True, position is treated as normalized (0.0=closed, 1.0=open)
                    If False, position is treated as absolute ticks
        """
        # Convert normalized to ticks if needed
        if normalized:
            position = self._normalized_to_ticks(position)
        else:
            position = int(position)
        
        with self._gripper_lock:
            self._gripper_cmd = position
            self._gripper_interrupt.set()

    def _gripper_loop(self):
        """Internal thread loop that executes gripper commands"""
        while True:
            self._gripper_interrupt.wait()
            with self._gripper_lock:
                cmd = self._gripper_cmd
                self._gripper_cmd = None
                self._gripper_interrupt.clear()
            if cmd:
                self._move_to_pos_blocking(cmd)  # Internal blocking call
    
    def _move_to_pos_blocking(self, goal_position):
        if self.dxl is None:
            print("Warning: Gripper DXL not initialized, cannot move to position")
            return
        self.dxl.move_to(goal_position)

        loops = 0
        prev_position = float('-inf')
        tot_loops = 0
        
        while True:
            dxl_present_position = self.dxl.get_present_position()

            if abs(dxl_present_position - prev_position) < 5 and tot_loops > MIN_TOTAL_LOOPS:
                loops += 1
                
            if loops > 5:
                self.dxl.move_to(dxl_present_position)
                break
            prev_position = dxl_present_position
            
            tot_loops += 1
        
        print("Desired position:%03d  Final position:%03d" % (goal_position, dxl_present_position))
    
    def is_ready(self):
        """Check if gripper is initialized and ready"""
        return self.dxl is not None
    
    def pull_status(self):
        """Update status dictionary with current position"""
        if self.dxl is None:
            return
        self.status = {
            'pos_ticks': self.dxl.get_present_position()
        }
    
    def home(self):
        """Home the gripper (move to zero position)"""
        if self.dxl is None:
            return
        zero_ticks = self.params['zero_t']
        self._move_to_pos_blocking(zero_ticks)
    
    def move_to(self, position, normalized=False):
        """
        Move to position.
        Args:
            position: Target position value
            normalized: If True, position is treated as normalized (0.0=closed, 1.0=open)
                    If False, position is treated as absolute ticks
        """
        if self.dxl is None:
            return
        
        # Convert normalized to ticks if needed
        if normalized:
            ticks = self._normalized_to_ticks(position)
        else:
            ticks = int(position)
        
        # Clamp to range if desired
        range_min = self.params['range_t'][0]
        range_max = self.params['range_t'][1]
        ticks_clamped = max(range_min, min(range_max, ticks))
        self._move_to_pos_blocking(ticks_clamped)
    
    def move_by(self, delta_ticks):
        """
        Move by relative amount in ticks.
        Direction depends on encoder polarity:
        - If encoder increases when opening: Positive = open, Negative = close
        - If encoder decreases when opening: Positive = close, Negative = open
        Args:
            delta_ticks: Relative movement in ticks (sign depends on encoder polarity)
        """
        if self.dxl is None:
            return
        self.pull_status()
        current_ticks = self.status['pos_ticks']
        new_ticks = current_ticks + delta_ticks
        print(f"Current ticks: {current_ticks}")
        print(f"Moving by {delta_ticks} ticks, new ticks: {new_ticks}")

        self._move_to_pos_blocking(new_ticks)
    
    def stop(self):
        """Hold current position"""
        if self.dxl is None:
            return
        current_pos = self.dxl.get_present_position()
        self.dxl.move_to(current_pos)
        time.sleep(0.1)  # Brief pause to let it settle
    
    def disable(self):
        """Disable the gripper"""
        if self.dxl:
            self.dxl.disable()
    
    def ticks_to_world_rad(self, ticks):
        """
        Stub conversion method for compatibility.
        Since we're using global ticks, this just returns ticks as-is.
        Can be implemented properly if needed for other interfaces.
        """
        return ticks  # Return ticks directly for now
    
    def world_rad_to_pct(self, world_rad):
        """
        Convert world_rad (or ticks) to percentage based on calibrated range.
        Since we're using global ticks, world_rad is treated as ticks.
        """
        # Treat world_rad as ticks
        ticks = int(world_rad)
        range_min = self.params['range_t'][0]
        range_max = self.params['range_t'][1]
        if range_max - range_min == 0:
            return 0.0
        pct = (ticks - range_min) / (range_max - range_min)
        return max(0.0, min(1.0, pct))
    
    def _load_params_from_config(self):
        """Load parameters from YAML config file"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'configs', 'gripper_calibration.yaml')
        
        if not os.path.exists(config_path):
            print(f"Gripper config not found at {config_path}, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Load stretch_gripper params if they exist
            if 'stretch_gripper' in config:
                sg = config['stretch_gripper']
                if 'zero_t' in sg:
                    self.params['zero_t'] = sg['zero_t']
                if 'range_t' in sg:
                    self.params['range_t'] = sg['range_t']
                if 'flip_encoder_polarity' in sg:
                    self.params['flip_encoder_polarity'] = sg['flip_encoder_polarity']
            
            print(f"Loaded gripper params from {config_path}")
            print(f"  zero_t: {self.params['zero_t']}")
            print(f"  range_t: {self.params['range_t']}")
        except Exception as e:
            print(f"Error loading gripper config: {e}, using defaults")
    
    def write_configuration_param_to_YAML(self, param_path, value):
        """
        Write configuration parameter to YAML file.
        param_path: e.g., 'stretch_gripper.range_t'
        value: value to write
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'configs', 'gripper_calibration.yaml')
        
        # Load existing config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error reading config: {e}, creating new config")
                config = {}
        else:
            config = {}
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Navigate/create nested structure
        keys = param_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
        
        # Update local params
        if keys[0] == 'stretch_gripper' and keys[1] in self.params:
            self.params[keys[1]] = value
        
        # Write back
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Wrote {param_path} = {value} to {config_path}")
        except Exception as e:
            print(f"Error writing config: {e}")
    
    def read_configuration_from_YAML(self, param_path, default=None):
        """Read configuration parameter from YAML file"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'configs', 'gripper_calibration.yaml')
        
        if not os.path.exists(config_path):
            return default
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Navigate nested structure
            keys = param_path.split('.')
            current = config
            for key in keys:
                if key not in current:
                    return default
                current = current[key]
            
            return current
        except Exception as e:
            print(f"Error reading config: {e}")
            return default

    def policy_setup(self, dxl_id=1, baudrate=None):
        # Get all available USB devices
        devices = (sorted(glob.glob("/dev/ttyUSB*")) + 
                sorted(glob.glob("/dev/ttyACM*")))
        
        # First, try to check for ID 1 with baudrate 57600
        test_id = 1
        test_baudrate = 57600
        found_new_motor = False
        
        print(f"Checking for new motor with ID {test_id} at baudrate {test_baudrate}...")
        
        for dev in devices:
            if not os.path.exists(dev):
                continue
            
            try:
                port = PortHandler(dev)
                pack = PacketHandler(PROTOCOL_VERSION)
                
                if not port.openPort():
                    continue
                if not port.setBaudRate(test_baudrate):
                    port.closePort()
                    continue
                
                time.sleep(0.01)
                
                # Try to ping ID 1
                try:
                    model, comm_res, dxl_err = pack.ping(port, test_id)
                except TypeError:
                    model, comm_res = pack.ping(port, test_id)
                    dxl_err = 0
                
                port.closePort()
                
                if comm_res == COMM_SUCCESS and dxl_err == 0 and model not in (None, 0):
                    found_new_motor = True
                    # Connect with the current motor settings (ID 1, baudrate 57600)
                    self.dxl = DXL(dev, PROTOCOL_VERSION, test_baudrate, test_id)
                    self.devicename = dev
                    print(f"  ✓ Found new motor @ {dev} | ID {test_id} | baudrate {test_baudrate} | model {model}")
                    break
                    
            except Exception as e:
                try:
                    port.closePort()
                except:
                    pass
                continue
        
        if not found_new_motor:
            print("  ✗ There is no new motor")
            print("Motor Setup Complete")
            return

        self.dxl.set_baudrate(BAUDRATE)
        self.dxl.set_ID(14)
        self.dxl.set_return_delay_time(0)
        self.dxl.enable_torque()
        self.dxl.set_pos_d_gain(0)
        self.dxl.set_profile_acceleration(21)
        self.dxl.set_profile_velocity(300)
        self.dxl.set_operating_mode(3)
        print("✓ Gripper setup is complete")