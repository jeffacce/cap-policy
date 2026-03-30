from .tensor_subscriber import TensorSubscriber
from .Franka import FrankaArm
from .deoxys_utils.franka_server import FrankaServer
from ..rpc import RPCServer
import atexit
import threading
import hydra
from omegaconf import DictConfig

import time
import zmq
from ..zmq_utils import *
    
class Listener(ProcessInstantiator):
    def __init__(self, franka, port_configs, franka_config=None):
        super().__init__()
        self.Franka = franka
        self.franka_config = franka_config or {}
        
        print("starting robot listener")
        print(port_configs)
        if self.Franka is None:
            # Get deoxys config from Hydra config or use default
            deoxys_config = self.franka_config.get("deoxys_config_path", "franka_config.yml")
            node_name = self.franka_config.get("node_name", "lambda")
            
            print(f"Using deoxys config: {deoxys_config}")
            print(f"Node name: {node_name}")
            
            # Start the FrankaServer in a separate thread to handle NUC communication
            self.franka_server = FrankaServer(deoxys_config)
            self.server_thread = threading.Thread(target=self.franka_server.init_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Wait a moment for the server to initialize
            time.sleep(2)

            self.robot = FrankaArm()

        self.robot.home() 
        self.server = RPCServer(self.robot, port_configs["host"], port_configs["action_port"])
    
    def stream(self):
        self.server.start()
        print("server started")
    
    @atexit.register
    def stop(self):
        self.server.stop()
        print("server stopped")
