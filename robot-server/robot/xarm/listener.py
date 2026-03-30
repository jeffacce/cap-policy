from .tensor_subscriber import TensorSubscriber
from .xarm import xArm
from ..rpc import RPCServer
import atexit

import time
import zmq
from ..zmq_utils import *
    
class Listener(ProcessInstantiator):
    def __init__(self, xarm, port_configs):
        super().__init__()
        self.xarm = xarm
        
        print("starting robot listner")
        print(port_configs)
        if self.xarm is None:
            self.robot = xArm(xarm_ip=port_configs["xarm_ip"])

        self.robot.home()
        self.server = RPCServer(self.robot, port_configs["host"], port_configs["action_port"])
    
    def stream(self):
        self.server.start()
        print("server started")
    
    @atexit.register
    def stop(self):
        self.server.stop()
        print("server stopped")
    