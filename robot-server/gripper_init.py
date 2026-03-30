from robot.gripper import Gripper

if __name__ == "__main__":
    gripper = Gripper(dxl_id=1, baudrate=57600)
    gripper.policy_setup()
