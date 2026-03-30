import hydra
from initializers import StartServer
import signal
import sys
import os

# Available controller options
CONTROLLER_OPTIONS = {
    'xarm': 'xArm robot controller',
    'franka': 'Franka robot controller', 
    'stretch': 'Hello-Stretch robot controller',
    'piper': 'Piper robot controller'
}

@hydra.main(config_path="configs", config_name="start_server", version_base="1.2")
def main(cfg):
    """
    Main function to start the server
    """
    #Start the server
    server = StartServer(cfg)
    processes = server.get_processes()

    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print("Server started")

    def signal_handler(sig, frame):
        for process in processes:
            process.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

def print_usage():
    """Print usage information for available controllers"""
    print("Robot Server - Available Controllers:")
    print("=====================================")
    print("Usage: python start_server.py [controller=<controller_name>] [HYDRA_OVERRIDES]")
    print()
    print("Available controllers:")
    for controller, description in CONTROLLER_OPTIONS.items():
        print(f"  {controller:<10} - {description}")
    print()
    print("Common Configuration Overrides:")
    print("  network.xarm_ip=<ip>                 # Set xArm IP address")
    print("  network.host_address=<ip>            # Set host address")
    print("  network.remote_address=<ip>          # Set remote address")
    print()
    print("Gripper Auto-Detection:")
    print("  The system automatically detects Dynamixel gripper devices on common USB ports.")
    print("  No manual configuration needed - just plug in your gripper!")
    print()
    print("Examples:")
    print("  python start_server.py                                    # Use default (xarm)")
    print("  python start_server.py controller=franka                  # Use Franka controller (right arm)")
    print("  python start_server.py controller=franka_left             # Use Franka controller (left arm)")
    print("  python start_server.py controller=position                # Use position controller for Hello-Stretch")
    print("  python start_server.py controller=piper                   # Use Piper robot controller")
    print()
    print("Franka Configuration Overrides:")
    print("  controller.franka_config.deoxys_config_path=deoxys_left.yml  # Use left arm Deoxys config")
    print("  controller.franka_config.nuc.ip=192.168.100.202            # Change NUC IP address")
    print("  controller.franka_config.robot.ip=172.16.1.4               # Change robot IP (left arm)")
    print("  controller.franka_config.control.policy_rate=30             # Change control frequency")
    print()
 

if __name__ == "__main__":
    # Check if user wants help
    if len(sys.argv) > 1 and any(arg in ['-h', '--help', 'help'] for arg in sys.argv):
        print_usage()
        sys.exit(0)
    
    # Check if user provided an invalid controller
    for arg in sys.argv:
        if arg.startswith('controller='):
            controller_name = arg.split('=')[1]
            if controller_name not in CONTROLLER_OPTIONS:
                print(f"Error: Invalid controller '{controller_name}'")
                print()
                print_usage()
                sys.exit(1)
    
    main()
